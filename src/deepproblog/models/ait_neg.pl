nn(net1, [X], Z, [benign, attack]) :: exphil(X, Z).
nn(net2, [X], Z, [benign, attack]) :: recon(X, Z). 
nn(net3, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net4, [X], Z, [benign, attack]) :: cracking(X, Z).

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
    exphil(X, attack).

multi_step(1, X, S, D, P, benign) :- 
    \+ multi_step(1, X, S, D, P, phase1).


multi_step(2, X, S, D, _, phase2) :-
    internal_traffic(S, D),
    recon(X, attack).
    
multi_step(2, X, S, D, _, benign) :-
    \+ multi_step(2, X, S, D, _, phase2).

    
multi_step(3, X, S, D, P, phase3) :- 
    internal_traffic(S, D),
    https(P),
    exploit(X, attack).

multi_step(3, X, S, D, P, benign) :-
    \+ multi_step(3, X, S, D, P, phase3).


multi_step(4, X, S, D, P, phase4) :-
    internal_traffic(S, D),
    (http(P);https(P)),
    cracking(X, attack).

multi_step(4, X, S, D, P, benign) :-
    \+ multi_step(4, X, S, D, P, phase4).