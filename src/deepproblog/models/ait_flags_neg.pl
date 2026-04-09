nn(net1, [X], Z, [benign, attack]) :: exphil(X, Z).
nn(net2, [X], Z, [benign, attack]) :: recon(X, Z). 
nn(net3, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net4, [X], Z, [benign, attack]) :: cracking(X, Z).

% Per phase rules

multi_step(1, X, phase1) :-
    exphil(X, attack).

multi_step(1, X, benign) :- 
    \+ multi_step(1, X, phase1).

multi_step(2, X, phase2) :-
    recon(X, attack).
    
multi_step(2, X, benign) :-
    \+ multi_step(2, X, phase2).
    
multi_step(3, X, phase3) :- 
    exploit(X, attack).

multi_step(3, X, benign) :-
    \+ multi_step(3, X, phase3).

multi_step(4, X, phase4) :-
    cracking(X, attack).

multi_step(4, X, benign) :-
    \+ multi_step(4, X, phase4).