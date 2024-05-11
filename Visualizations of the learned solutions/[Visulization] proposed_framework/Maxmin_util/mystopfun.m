function stopnow = mystopfun(problem, x, info, last)
    % https://www.manopt.org/reference/manopt/core/stoppingcriterion.html


    % This tells the solver to exit as soon as two successive iterations combined have decreased the cost by less than 10-3.
    % stopnow = (last >= 3 && info(last-2).cost - info(last).cost < 1e-3);
    
    % as Tao suggest in his paper, the subgradient method cannot guarantee decrease in the objective function for each iteration
    % thus, we iterate the RCG for sufficient times, e.g., 10000 iterations
    
    stopnow = (last >= 10000 && info(last-2).cost - info(last).cost < 1e-5);
    
end