using Knet

True=true 
include("common/params.py"); #hyperparams

#system specs
println("OS: ", Sys.KERNEL)
println("Julia: ", VERSION)
println("Knet: ", Pkg.installed("Knet"))
println("GPU: ", readstring(`nvidia-smi --query-gpu=name --format=csv,noheader`))


function initmodel(; atype=KnetArray, dtype=Float32, winit=xavier, binit=zeros)
    w(dims...)=atype(winit(dtype,dims...))
    b(dims...)=atype(binit(dtype,dims...))
    return Any[
        w(3,3,3,50), b(1,1,50,1),
        w(3,3,50,50), b(1,1,50,1),
        w(3,3,50,100), b(1,1,100,1),
        w(3,3,100,100), b(1,1,100,1),
        w(512,6400), b(512,1),
        w(10,512), b(10,1)
    ]
end

function predict(w,x)
	
    convbias(x,w,b) = conv4(w,x;padding=1) .+ b
    fc(x,w,b) = w * mat(x) .+ b;
    x = relu.(convbias(x,w[1],w[2]))
    x = relu.(pool(convbias(x,w[3],w[4])))
    x = dropout(x,0.25)
    x = relu.(convbias(x,w[5],w[6]))
    x = relu.(pool(convbias(x,w[7],w[8])))
    x = dropout(x,0.25)
    x = relu.(fc(x,w[9],w[10]))
    x = dropout(x,0.5)
    return fc(x,w[11],w[12])
end

loss(w,x,y)=nll(predict(w,x),y) # nll: negative log likelihood
lossgradient = grad(loss);

function gradOnWorker(gpuID,w,x,y)
    gpu(gpuID)
    return lossgradient(w,x,y)
end

#minibatch functions
function convertYs(ys,bs,s)
    yCont = Any[]
    for i=1:bs:s
        if i+bs-1<s
            push!(yCont,ys[i:i+bs-1])
        else
            push!(yCont,ys[i:s])
        end
    end
    return yCont
end

function convertXs(xs,bs,s)
    xCont = Any[]
    for i=1:bs:s
        if i+bs-1 < s
            p = KnetArray(xs[:,:,:,i:i+bs-1])
        else
            p = KnetArray(xs[:,:,:,i:s])
        end
        push!(xCont,p)
    end

    return xCont
end

# load data
include(Knet.dir("data","cifar.jl"))
@time (xtrn,ytrn,xtst,ytst,lbls)=cifar10()
for d in (xtrn,ytrn,xtst,ytst); println(summary(d)); end

#divide data among workers
gpu(1)
x2=convertXs(xtrn[:,:,:,1:25000],floor(Int64,BATCHSIZE/2),25000)
y2=convertYs(ytrn[1:25000],floor(Int64,BATCHSIZE/2),25000)

gpu(2)
x3=convertXs(xtrn[:,:,:,25001:50000],floor(Int64,BATCHSIZE/2),25000)
y3=convertYs(ytrn[25001:50000],floor(Int64,BATCHSIZE/2),25000)



# prepare for training
gpu(0)
model = optim = nothing; knetgc() # Clear memory from last run
model = initmodel()
optim = optimizers(model, Momentum; lr=LR, gamma=MOMENTUM);

#grads container for averaging on cpu
gradsGlob = Array{Any}(12)
#enable GPU direct
Knet.enableP2P()

# 159s
info("Training...")
#calculate batch count
bCount = floor(Int64,25000/BATCHSIZE/2)

#intiialize GPU Arrays on worker GPUs
gpu(1)
g2 = initmodel()

gpu(2)
g3 = initmodel()

models = Array{Any}(2)
models[1] = g2
models[2] = g3
#training loop
@time for epoch in 1:EPOCHS
	@time for iter=1:bCount #for each minibatch
		
		    Threads.@threads for tid=1:Threads.nthreads()
                #switch to worker gpu
                #gpu(tid)
                
                models[tid] = gradOnWorker(tid,models[tid],x2[iter],y2[iter])
                
            end

			gpu(0) #switch to master gpu

			#fetch remote references from workers
			#g12 = @getfrom 2 g2 
			#g13 = @getfrom 3 g3 


			#take the mean on CPU
			for i = 1:12
				gradsGlob[i] = models[1][i]./2 + models[2][i]./2
			end

			#grads = map(a->convert(KnetArray{Float32},a),gradsGlob) #convert to KnetArray

			update!(model, gradsGlob, optim) #update the global model
			
            #copy updated model back to GPUs
            gpu(1)
            g2=copy(model)
            gpu(2)
            g3=copy(model)
    end
end