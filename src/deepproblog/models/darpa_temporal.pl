%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Phase Detectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nn(net1, [X], Z, [benign, phase1]) :: recon_det(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: ping_det(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: overflow_det(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: install_det(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos_det(X, Z).

step(1,X,P1,P2,P3,P4,Outcome) :-
    recon_det(X, Outcome).

step(2,X,P1,P2,P3,P4,Outcome) :-
    P1 = 1,
    ping_det(X, Outcome).

step(3,X,P1,P2,P3,P4,Outcome) :-
    P1 = 1,
    P2 = 1,
    overflow_det(X, Outcome).

step(4,X,P1,P2,P3,P4,Outcome) :-
    P1 = 1,
    P2 = 1,
    P3 = 1,
    install_det(X, Outcome).

step(5,X,P1,P2,P3,P4,Outcome) :-
    P1 = 1,
    P2 = 1,
    P3 = 1,
    P4 = 1,
    ddos_det(X, Outcome).


multi_step_attack(X,P1,P2,P3,P4,Outcome) :-
    step(5,X,P1,P2,P3,P4,Outcome).

multi_step_attack(_,_,_,_,_,benign).