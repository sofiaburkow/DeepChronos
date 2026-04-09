nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

multi_step(1, X, phase1) :-
    msa(X, attack).

multi_step(1, X, benign) :- 
    \+ multi_step(1, X, phase1).

multi_step(2, X, phase2) :-
    msa(X, attack).
    
multi_step(2, X, benign) :-
    \+ multi_step(2, X, phase2).

multi_step(3, X, phase3) :- 
    msa(X, attack).

multi_step(3, X, benign) :-
    \+ multi_step(3, X, phase3).

multi_step(4, X, phase4) :-
    msa(X, attack).

multi_step(4, X, benign) :-
    \+ multi_step(4, X, phase4).