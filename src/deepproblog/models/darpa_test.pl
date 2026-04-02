nn(net1, [X], Z, [benign, attack]) :: ping(X, Z).
nn(net2, [X], Z, [benign, attack]) :: probing(X, Z). 
nn(net3, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net4, [X], Z, [benign, attack]) :: install(X, Z).
nn(net5, [X], Z, [benign, attack]) :: ddos(X, Z).

% Network direction 
home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

% Protocol 
icmp(1).
tcp(6).
udp(17).

% Vulnerability knowledge

sadmind_port(111).

telnet(23).
rsh(514).

privileged_port(1020).
privileged_port(1021).
privileged_port(1022).
privileged_port(1023).
% privileged_port(P) :- P < 1024.

http(80).

% Traffic pattern rules

icmp_req(SO, DO, Proto) :-
    ext_orig(SO),
    home_resp(DO),
    icmp(Proto).

icmp_resp(SO, DO, Proto) :-
    home_orig(SO),
    ext_resp(DO),
    icmp(Proto).

udp_req(SO, DO, Proto) :-
    ext_orig(SO),
    home_resp(DO),
    udp(Proto).

tcp_req(SO, DO, Proto) :-
    ext_orig(SO),
    home_resp(DO),
    tcp(Proto).

% Multi-step attack logic

% === Phase 1 logic ===

phase(1, X, SO, DO, _, Proto, _, _, phase1) :-
    icmp_req(SO, DO, Proto),
    ping(X, attack).

phase(1, X, SO, DO, _, Proto, _, _, benign) :- 
    \+ phase(1, X, SO, DO, _, Proto, _, _, phase1).


% === Phase 2 logic ===

t(1.0)::phase2_signal(SO, DO, DPort, Proto) :-
    udp_req(SO, DO, Proto),
    sadmind_port(DPort).

t(1.0)::phase2_signal(SO, DO, _, Proto) :-
    icmp_resp(SO, DO, Proto).

t(0.8) :: phase2_signal(SO, DO, DPort, Proto) :-
    udp_req(SO, DO, Proto),
    \+ sadmind_port(DPort).

phase(2, X, SO, DO, DPort, Proto, _, _, phase2) :-
    phase2_signal(SO, DO, DPort, Proto),
    probing(X, attack).

phase(2, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(2, X, SO, DO, DPort, Proto, _, _, phase2).


% === Phase 3 logic ===

t(1.0)::phase3_signal(SO, DO, DPort, Proto) :-
    udp_req(SO, DO, Proto),
    sadmind_port(DPort).

t(1.0)::phase3_signal(SO, DO, DPort, Proto) :-
    udp_req(SO, DO, Proto),
    \+ sadmind_port(DPort).

t(1.0)::phase3_signal(SO, DO, DPort, Proto) :-
    tcp_req(SO, DO, Proto),
    telnet(DPort).

phase(3, X, SO, DO, DPort, Proto, _, _, phase3) :-
    phase3_signal(SO, DO, DPort, Proto),
    exploit(X, attack).

phase(3, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(3, X, SO, DO, DPort, Proto, _, _, phase3).


% === Phase 4 logic ===

phase4_signal(SO, DO, DPort, Proto) :-
    tcp_req(SO, DO, Proto),
    telnet(DPort).

phase4_signal(_, _, DPort, Proto) :-
    (rsh(DPort); privileged_port(DPort)),
    tcp(Proto).

phase(4, X, SO, DO, DPort, Proto, _, _, phase4) :-
    phase4_signal(SO, DO, DPort, Proto),
    install(X, attack).

phase(4, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(4, X, SO, DO, DPort, Proto, _, _, phase4).

% === Phase 5 logic ===

t(1.0)::phase5_signal(1).

phase(5, X, _, _, _, _, _, DS, phase5) :-
    phase5_signal(DS),
    ddos(X, attack).

phase(5, X, _, _, _, _, _, DS, benign) :-
    \+ phase(5, X, _, _, _, _, _, DS, phase5).

% Overall multi-step attack logic

multi_step(Next, X, SO, DO, DPort, Proto, _, DS, Outcome) :-
    phase(Next, X, SO, DO, DPort, Proto, _, DS, Outcome). 