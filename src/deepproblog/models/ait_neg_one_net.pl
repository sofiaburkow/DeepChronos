nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

% Valid phase progressions

attack_phase(phase1).
attack_phase(phase2).
attack_phase(phase3).
attack_phase(phase4).

valid_phase_progression(P1,P2,P3,Compromised,phase1) :- P1 = 0, P2 = 0, P3 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,Compromised,phase2) :- P1 = 1, P2 = 0, P3 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,Compromised,phase3) :- P1 = 1, P2 = 1, P3 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,Compromised,phase4) :- Compromised = 1.

% Grounding facts

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

dns(53).
http(80).
https(443).

tcp(6).

internal_traffic(Src, Dst) :-
    home_orig(Src),
    home_resp(Dst).

exfil_signal(1).
scan_signal(1).

% Phase rules

phase_rule(X,Src,Dst,Port,Proto,ExSig,ScanSig,phase1) :-
    internal_traffic(Src,Dst),
    tcp(Proto),
    scan_signal(ScanSig).

phase_rule(X,Src,Dst,Port,Proto,ExSig,ScanSig,phase2) :-
    internal_traffic(Src,Dst),
    https(Port).

phase_rule(X,Src,Dst,Port,Proto,ExSig,ScanSig,phase3) :-
    tcp(Proto),
    (http(Port);https(Port)).

phase_rule(X,Src,Dst,Port,Proto,ExSig,ScanSig,phase4) :-
    internal_traffic(Src,Dst),
    dns(Port),
    exfil_signal(ExSig).

% Multi-step attack definition

multi_step(X,P1,P2,P3,Compromised,Src,Dst,Port,Proto,ExSig,ScanSig,Phase) :-
    valid_phase_progression(P1,P2,P3,Compromised,Phase),
    phase_rule(X,Src,Dst,Port,Proto,ExSig,ScanSig,Phase),
    msa(X,attack).

multi_step(X,P1,P2,P3,Compromised,Src,Dst,Port,Proto,ExSig,ScanSig,benign) :-
    \+ (
        attack_phase(Phase),
        multi_step(X,P1,P2,P3,Compromised,Src,Dst,Port,Proto,ExSig,ScanSig,Phase)
    ).