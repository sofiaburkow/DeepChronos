% Neural networks

nn(net1, [X], Z, [benign, phase1]) :: recon(X, Z).
nn(net2, [X], Z, [benign, phase2]) :: ping(X, Z).
nn(net3, [X], Z, [benign, phase3]) :: overflow(X, Z).
nn(net4, [X], Z, [benign, phase4]) :: install(X, Z).
nn(net5, [X], Z, [benign, phase5]) :: ddos(X, Z).

% Expert knowledge

icmp(1).
tcp(6).
udp(17).

loc_orig(1).
loc_resp(1).
ext_orig(0).
ext_resp(0).

common_install_port(80).
common_install_port(443).
common_install_port(21).
common_install_port(23).
common_install_port(8080).
common_install_port(8000).

known_install_port(23).
known_install_port(514).
known_install_port(1022).

install_port(P) :- known_install_port(P).
% install_port(P) :- common_install_port(P).

% Vulnerability knowledge

sadmind_known_port(111).
sadmind_known_port(Port) :- Port >= 32771.
sadmind_port(P) :- sadmind_known_port(P).

sadmind_followup_port(23).

% DARPA attack phase rules

icmp_resp(SO, DO, Proto) :-
    loc_orig(SO),
    ext_resp(DO),
    icmp(Proto).

sadmind_ping(SO, DO, DPort, Proto) :-
    ext_orig(SO),
    loc_resp(DO),
    udp(Proto),
    sadmind_port(DPort).

sadmind_exp(SO, DO, DPort, Proto) :-
    ext_orig(SO),
    loc_resp(DO),
    tcp(Proto),
    sadmind_followup_port(DPort).

install_signal(DPort, Proto) :-
    tcp(Proto),
    install_port(DPort).


% Multi-step attack logic

phase(1, X, SO, DO, _, Proto, _, _, phase1) :-
    ext_orig(SO),
    loc_resp(DO),
    % icmp(Proto),
    recon(X, phase1).

phase(1, X, SO, DO, _, Proto, _, _, benign) :- 
    \+ phase(1, X, SO, DO, _, Proto, _, _, phase1).


phase(2, X, SO, DO, DPort, Proto, _, _, phase2) :-
    sadmind_req(SO, DO, DPort, Proto),
    ping(X, phase2).

phase(2, X, SO, DO, DPort, Proto, _, _, phase2) :-
    icmp_resp(SO, DO, Proto),
    ping(X, phase2).

phase(2, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(2, X, SO, DO, DPort, Proto, _, _, phase2).
    

phase(3, X, SO, DO, DPort, _, _, _, phase3) :- 
    sadmind_req(SO, DO, DPort, Proto),
    overflow(X, phase3).

phase(3, X, SO, DO, DPort, _, _, _, phase3) :- 
    sadmind_exp(SO, DO, DPort, Proto),
    overflow(X, phase3).

phase(3, X, SO, DO, DPort, _, _, _, benign) :-
    \+ phase(3, X, SO, DO, DPort, _, _, _, phase3).


phase(4, X, _, _, DPort, Proto, _, _, phase4) :-
    install_signal(DPort, Proto),
    install(X, phase4).

phase(4, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(4, X, _, _, DPort, Proto, _, _, phase4).


phase(5, X, _, _, _, _, R, S, phase5) :-
    R > 0.5, S > 5,
    ddos(X, phase5).

phase(5, X, _, _, _, _, R, S, benign) :-
    \+ phase(5, X, _, _, _, _, R, S, phase5).


multi_step(X, Next, SO, DO, DPort, Proto, R, S, Outcome) :-
    phase(Next, X, SO, DO, DPort, Proto, R, S, Outcome). 