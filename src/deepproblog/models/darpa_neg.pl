nn(net1, [X], Z, [benign, attack]) :: ping(X, Z).
nn(net2, [X], Z, [benign, attack]) :: recon(X, Z).
nn(net3, [X], Z, [benign, attack]) :: overflow(X, Z).
nn(net4, [X], Z, [benign, attack]) :: install(X, Z).
nn(net5, [X], Z, [benign, attack]) :: ddos(X, Z).

icmp(1).
tcp(6).
udp(17).

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

sadmind_known_port(111).
sadmind_known_port(Port) :- Port >= 32771.
sadmind_port(P) :- sadmind_known_port(P).

known_install_port(23).
known_install_port(514).
known_install_port(1022).
install_port(P) :- known_install_port(P).

% Attack phase rules

icmp_req(SO, DO, Proto) :-
    ext_orig(SO),
    home_resp(DO),
    icmp(Proto).

icmp_resp(SO, DO, Proto) :-
    home_orig(SO),
    ext_resp(DO),
    icmp(Proto).

sadmind_req(SO, DO, DPort, Proto) :-
    udp(Proto),
    ext_orig(SO),
    home_resp(DO),
    sadmind_port(DPort).

% Multi-step attack logic

phase(1, X, SO, DO, _, Proto, _, _, phase1) :-
    icmp_req(SO, DO, Proto),
    ping(X, attack).

phase(1, X, SO, DO, _, Proto, _, _, benign) :- 
    \+ phase(1, X, SO, DO, _, Proto, _, _, phase1).
    

phase(2, X, SO, DO, DPort, Proto, _, _, phase2) :-
    sadmind_req(SO, DO, DPort, Proto),
    recon(X, attack).

phase(2, X, SO, DO, _, Proto, _, _, phase2) :-
    icmp_resp(SO, DO, Proto),
    recon(X, attack).

phase(2, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(2, X, SO, DO, DPort, Proto, _, _, phase2).


phase(3, X, SO, DO, DPort, Proto, _, _, phase3) :- 
    sadmind_req(SO, DO, DPort, Proto),
    overflow(X, attack).

phase(3, X, SO, DO, _, Proto, _, _, phase3) :-
    udp(Proto),
    ext_orig(SO),
    home_resp(DO),
    overflow(X, attack).

phase(3, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(3, X, SO, DO, DPort, Proto, _, _, phase3).


phase(4, X, _, _, DPort, Proto, _, _, phase4) :-
    tcp(Proto),
    install_port(DPort),
    install(X, attack).

phase(4, X, _, _, DPort, Proto, _, _, benign) :-
    \+ phase(4, X, _, _, DPort, Proto, _, _, phase4).


ddos_evidence(R, S) :- R > 0.2, S > 2.

phase(5, X, _, _, _, _, R, S, phase5) :-
    ddos_evidence(R, S),
    ddos(X, attack).

phase(5, X, _, _, _, _, R, S, benign) :-
    \+ phase(5, X, _, _, _, _, R, S, phase5).


multi_step(X, Next, SO, DO, DPort, Proto, R, S, Outcome) :-
    phase(Next, X, SO, DO, DPort, Proto, R, S, Outcome). 