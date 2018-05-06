#using Knet
@everywhere function initmodel(gpuID; atype=KnetArray, dtype=Float32, winit=xavier, binit=zeros)
	gpu(gpuID)
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
    #return g
end;


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
