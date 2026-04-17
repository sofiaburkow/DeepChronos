nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

% ---------------------------
% Grounding facts
% ---------------------------

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

% ---------------------------
% Signals
% ---------------------------

exfil_signal(1).
scan_signal(1).

% ---------------------------
% Phase definitions
% ---------------------------

phase(phase1).
phase(phase2).
phase(phase3).
phase(phase4).

% ---------------------------
% Valid phase transitions
% ---------------------------

t(0.6)::valid_phase(0,0,0,0,phase1).
t(1.0)::valid_phase(1,0,0,0,phase1).

t(1.0)::valid_phase(1,1,0,0,phase2).
t(1.0)::valid_phase(1,1,1,0,phase3).
t(1.0)::valid_phase(1,1,1,1,phase4).

% ---------------------------
% Phase rules (cleaned)
% ---------------------------

phase_rule(Src,Dst,Port,Proto,_,_,phase1) :-
    internal_traffic(Src,Dst),
    dns(Port).

phase_rule(Src,Dst,Port,Proto,_,_,phase2) :-
    internal_traffic(Src,Dst),
    tcp(Proto).

phase_rule(Src,Dst,Port,Proto,_,_,phase3) :-
    internal_traffic(Src,Dst),
    https(Port).

phase_rule(_,_,Port,Proto,_,_,phase4) :-
    tcp(Proto),
    (http(Port); https(Port)).

% ---------------------------
% Multi-step attack
% ---------------------------

multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,ExSig,ScanSig,Phase) :-
    phase(Phase),
    valid_phase(P1,P2,P3,P4,Phase),
    msa(X,attack),
    phase_rule(Src,Dst,Port,Proto,ExSig,ScanSig,Phase).

% ---------------------------
% Benign definition (optimized)
% ---------------------------

multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,ExSig,ScanSig,benign) :-
    \+ (
        phase(Phase),
        multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,ExSig,ScanSig,Phase)
    ).