nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

dns(53).
http(80).
https(443).

internal_traffic(S, D) :-
    home_orig(S),
    home_resp(D).


multi_step(1, X, S, D, P, phase1) :-
    internal_traffic(S, D),
    dns(P),
    msa(X, attack).

multi_step(1, X, S, D, P, benign) :- 
    \+ multi_step(1, X, S, D, P, phase1).


multi_step(2, X, S, D, _, phase2) :-
    internal_traffic(S, D),
    msa(X, attack).
    
multi_step(2, X, S, D, _, benign) :-
    \+ multi_step(2, X, S, D, _, phase2).

    
multi_step(3, X, S, D, P, phase3) :- 
    internal_traffic(S, D),
    https(P),
    msa(X, attack).

multi_step(3, X, S, D, P, benign) :-
    \+ multi_step(3, X, S, D, P, phase3).


multi_step(4, X, S, D, P, phase4) :-
    internal_traffic(S, D),
    (http(P);https(P)),
    msa(X, attack).

multi_step(4, X, S, D, P, benign) :-
    \+ multi_step(4, X, S, D, P, phase4).