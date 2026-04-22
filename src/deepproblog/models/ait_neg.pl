nn(net1, [X], Z, [benign, attack]) :: scan(X, Z).
nn(net2, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net3, [X], Z, [benign, attack]) :: priv_esc(X, Z).
nn(net4, [X], Z, [benign, attack]) :: data_exfil(X, Z).

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

% Ports
dns(53).
http(80).
https(443).

% Protocols
tcp(6).
udp(17).

% Services
no_service(0).
http(1).
https(2).
ssl(3).

internal_traffic(Src, Dst) :-
    home_orig(Src),
    home_resp(Dst).

% t(1.0)::exfil_signal(1).
t(1.0)::scan_signal(1).

% Phase rules

phase_rule(X,Src,Dst,Port,Proto,Service,S,E,phase1) :-
    internal_traffic(Src,Dst),
    tcp(Proto), 
    % scan_signal(FOR),
    scan(X,attack).

phase_rule(X,Src,Dst,Port,Proto,Service,S,E,phase2) :-
    internal_traffic(Src,Dst),
    https(Port),
    exploit(X,attack).

phase_rule(X,Src,Dst,Port,Proto,Service,S,E,phase3) :-
    tcp(Proto),
    (http(Port);https(Port)),
    priv_esc(X,attack).

phase_rule(X,Src,Dst,Port,Proto,Service,S,E,phase4) :-
    % internal_traffic(Src,Dst),
    % udp(Proto),
    dns(Port),
    E = 1,
    data_exfil(X,attack).

% Multi-step attack definition

multi_step(X,P1,P2,P3,Compromised,Src,Dst,Port,Proto,Service,S,E,Phase) :-
    valid_phase_progression(P1,P2,P3,Compromised,Phase),
    phase_rule(X,Src,Dst,Port,Proto,Service,S,E,Phase).

multi_step(X,P1,P2,P3,Compromised,Src,Dst,Port,Proto,Service,S,E,benign) :-
    \+ (
        attack_phase(Phase),
        multi_step(X,P1,P2,P3,Compromised,Src,Dst,Port,Proto,Service,S,E,Phase)
    ).