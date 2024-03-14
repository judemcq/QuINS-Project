function x = StateFcn(x)
    dt = 0.05;
    x = x + StateFcnContinuous(x)*dt;
end

function dxdt = StateFcnContinuous(x)
    dxdt = [x(2); (1-x(1)^2)*x(2)-x(1)];
end
    
    