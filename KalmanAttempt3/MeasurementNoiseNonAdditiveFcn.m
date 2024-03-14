function yk = MeasurementNoiseNonAdditiveFcn(xk,vk)
    yk = xk(1)*(1+vk);
end