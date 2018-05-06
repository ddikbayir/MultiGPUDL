#Working distributed model
#Execution: julia -p 2 driver.jl

@everywhere using Knet

#Nice package to do multi-process communication on CPU (KnetArray pointers disappear for some strange reason)
#see also: https://github.com/denizyuret/Knet.jl/issues/281

@everywhere using ParallelDataTransfer

@everywhere True=true 
@everywhere include("common/params.py"); #hyperparams
#@everywhere include("finalMod.jl"); #worker functions

#system specs
println("OS: ", Sys.KERNEL)
println("Julia: ", VERSION)
println("Knet: ", Pkg.installed("Knet"))
println("GPU: ", readstring(`nvidia-smi --query-gpu=name --format=csv,noheader`))


'''
@everywhere function initmodel(; atype=KnetArray, dtype=Float32, winit=xavier, binit=zeros)
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
'''
@everywhere function predict(w,x)
	
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

@everywhere loss(w,x,y)=nll(predict(w,x),y) # nll: negative log likelihood
@everywhere lossgradient = grad(loss);

#minibatch functions
@everywhere function convertYs(ys,bs,s)
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

@everywhere function convertXs(xs,bs,s)
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
#send data to workers
sendto(2,xtrn=xtrn[:,:,:,1:25000],ytrn=ytrn[1:25000])
sendto(3,xtrn=xtrn[:,:,:,25001:50000],ytrn=ytrn[25001:50000])

#divide data among workers
@everywhere gpu(1)
@defineat 2 x2=convertXs(xtrn,floor(Int64,BATCHSIZE/2),25000)
@defineat 2 y2=convertYs(ytrn,floor(Int64,BATCHSIZE/2),25000)

@everywhere gpu(2)
@defineat 3 x3=convertXs(xtrn,floor(Int64,BATCHSIZE/2),25000)
@defineat 3 y3=convertYs(ytrn,floor(Int64,BATCHSIZE/2),25000)



# prepare for training
@everywhere gpu(0)
model = optim = nothing; knetgc() # Clear memory from last run
model = initmodel()
optim = optimizers(model, Momentum; lr=LR, gamma=MOMENTUM);

#grads container for averaging on cpu
gradsGlob = Array{Any}(12)
#enable GPU direct
@everywhere Knet.enableP2P()

# 159s
info("Training...")
#calculate batch count
bCount = floor(Int64,25000/BATCHSIZE/2)

#intiialize GPU Arrays on worker GPUs
@everywhere gpu(1)
@defineat 2 g2 = initmodel()

@everywhere gpu(2)
@defineat 3 g3 = initmodel()

#training loop
@time for epoch in 1:EPOCHS
	@time for iter=1:bCount #for each minibatch
		
		@sync for (id, pid) in enumerate(workers()) # for each worker sync at end of the loop
			gpu(pid-1)
			cpuW = map(a->convert(Array{Float32},a), model)
			#send params to workers
			if pid == 2
				sendto(pid,w2=cpuW)
			elseif pid == 3
				sendto(pid,w3=cpuW)
			end

			@async begin #calculate grads in parallel
				#println(id, iter)
				
				if pid == 2
					gpu(pid-1) #switch to worker GPU
					@spawnat 2 wg2=map(a->convert(KnetArray{Float32},a),w2)
					@spawnat 2 g2 = lossgradient(wg2,x2[iter],y2[iter])
					@spawnat 2 map!(a->convert(Array{Float32},a), g2) #convert to CPU array since KnetArray pointers are reset during multi-process communication
				elseif pid == 3
					gpu(pid-1) #switch to worker GPU
					@spawnat 2 wg3=map(a->convert(KnetArray{Float32},a),w3)
					@spawnat 3 g3 = lossgradient(wg3,x3[iter],y3[iter])
					@spawnat 3 map!(a->convert(Array{Float32},a), g3) #convert to CPU array since KnetArray pointers are reset during multi-process communication
				end
				
			end
			
		end
			@everywhere gpu(0) #switch to master gpu

			#fetch remote references from workers
			g12 = @getfrom 2 g2 
			g13 = @getfrom 3 g3 


			#take the mean on CPU
			for i = 1:12
				gradsGlob[i] = g12[i]./2 + g13[i]./2
			end

			grads = map(a->convert(KnetArray{Float32},a),gradsGlob) #convert to KnetArray

			update!(model, grads, optim) #update the global model
			
    end
end