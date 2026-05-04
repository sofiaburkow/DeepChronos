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

% Network traffic pattern definitions

home_orig(1).
home_resp(1).

tcp(6).

dns_port(53).
http_port(80).
https_port(443).

int_to_int(SrcO, DstO) :- home_orig(SrcO), home_resp(DstO).

% Attack phase inference

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase1) :- int_to_int(SrcO,DstO), dns_port(DPort).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase1).

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase2) :- int_to_int(SrcO,DstO), tcp(Proto).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase2).

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase3) :- int_to_int(SrcO,DstO), https_port(DPort).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase3).

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase4) :- int_to_int(SrcO,DstO), tcp(Proto), (http_port(DPort) ; https_port(DPort)).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase4).

multi_step(X,P1,P2,P3,SrcO,DstO,DPort,Proto,NextPhase) :-
    next_attack_phase(P1,P2,P3,NextPhase),
    phase_soft(SrcO,DstO,DPort,Proto,NextPhase),
    event(X, malicious).

multi_step(X,P1,P2,P3,SrcO,DstO,DPort,Proto,benign) :-
    \+ (
        attack_phase(NextPhase),
        multi_step(X,P1,P2,P3,SrcO,DstO,DPort,Proto,NextPhase)
    ).