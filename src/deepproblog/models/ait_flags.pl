nn(net1, [X], Z, [benign, attack]) :: exphil(X, Z).
nn(net2, [X], Z, [benign, attack]) :: recon(X, Z). 
nn(net3, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net4, [X], Z, [benign, attack]) :: cracking(X, Z).

exphil_flag(1, 0, 0, 0).
recon_flag(1, 1, 0, 0).
exploit_flag(1, 1, 1, 0).
cracking_flag(1, 1, 1, 1).

multi_step(P1, P2, P3, P4, X, phase1) :-
    exphil(X, attack), 
    exphil_flag(P1, P2, P3, P4).

multi_step(P1, P2, P3, P4, X, phase2) :-
    recon(X, attack),
    recon_flag(P1, P2, P3, P4).

multi_step(P1, P2, P3, P4, X, phase3) :- 
    exploit(X, attack),
    exploit_flag(P1, P2, P3, P4).

multi_step(P1, P2, P3, P4, X, phase4) :-
    cracking(X, attack),
    cracking_flag(P1, P2, P3, P4).

multi_step(P1,P2,P3,P4,X,benign) :-
    \+ multi_step(P1,P2,P3,P4,X,phase1),
    \+ multi_step(P1,P2,P3,P4,X,phase2),
    \+ multi_step(P1,P2,P3,P4,X,phase3),
    \+ multi_step(P1,P2,P3,P4,X,phase4).