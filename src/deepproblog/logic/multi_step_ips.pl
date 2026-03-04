% Neural networks for each phase
nn(net1, [X], Z, [benign, phase1]) :: phase(1, X, Z).
nn(net2, [X], Z, [benign, phase2]) :: phase(2, X, Z).
nn(net3, [X], Z, [benign, phase3]) :: phase(3, X, Z).
nn(net4, [X], Z, [benign, phase4]) :: phase(4, X, Z).
nn(net5, [X], Z, [benign, phase5]) :: phase(5, X, Z).

% IP logic\
internal_ip(IP) :=
    IP in "172.16.*.*",
    "internal".
    
external_ip(IP) :=
    \+ internal_ip(IP),
    "external".

% recon and explotation
recon(SIP, DIP) :-
    external_ip(SIP),
    internal_ip(DIP).

% lateral movement
lateral_movement(SIP, DIP) :-
    internal_ip(SIP),
    internal_ip(DIP).

% final objective
final_objective(SIP, DIP) :-
    internal_ip(SIP),
    external_ip(DIP).

% Define multi-step logic program
multi_step_ips(X, SIP, DIP, P1, P2, P3, P4, Outcome) :-
    Next is P1 + P2 + P3 + P4 + 1,
    phase(Next, X, Outcome).