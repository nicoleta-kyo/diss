function [X,cvx_status] = cvx_opt_fim(d,S,D, hatF_hvec, R, v_theta)

% explicitly declare variables for cvx program in order to avoid dynamic
% assignment (and get static workspace error)
cvx_problem = []
cvx_status = []
cvx_cputime = []
cvx_optbnd = []
cvx_optval = []
cvx_slvitr = []
cvx_slvtol = []

% do cvx opt  
cvx_begin
    variable X( d, d ) symmetric
    minimize( sum_square_abs(R*F_to_Fhvec(d, X, D, S) - v_theta) );
    for k = 1:d,
        X(k,k) == hatF_hvec(k) ;
    end
     X == semidefinite(d);
cvx_end

function Fhvec = F_to_Fhvec(d, F, D, S)
   
    %create Fvec
    Fvec = reshape(F, [d*d,1])
    %create Fhvec
    Fhvec = pinv(S)*pinv(D)*Fvec

end
end