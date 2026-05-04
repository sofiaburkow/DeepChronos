nn(net1, [X], Z, [benign, malicious]) :: event(X, Z).

% Attack phase definitions

attack_phase(phase1).
attack_phase(phase2).
attack_phase(phase3).
attack_phase(phase4).

next_attack_phase(P1,P2,P3,phase1) :- P1 = 0, P2 = 0, P3 = 0.
next_attack_phase(P1,P2,P3,phase2) :- P1 = 1, P2 = 0, P3 = 0.
next_attack_phase(P1,P2,P3,phase3) :- P1 = 1, P2 = 1, P3 = 0.
next_attack_phase(P1,P2,P3,phase4) :- P1 = 1, P2 = 1, P3 = 1.

% Attack phase inference

multi_step(X,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,NextPhase) :-
    next_attack_phase(P1,P2,P3,NextPhase),
    event(X, malicious).

multi_step(X,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,benign) :-
    \+ (
        attack_phase(NextPhase),
        multi_step(X,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,NextPhase)
    ).