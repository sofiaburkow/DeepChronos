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

% download_port(80).
% download_port(443).
% download_port(21).
download_port(23).
% download_port(8080).
% download_port(8000).

% Vulnerability knowledge

sadmind_known_port(111).
sadmind_known_port(Port) :- Port >= 32771.
sadmind_port(P) :- sadmind_known_port(P).
sadmind_followup_port(23).

c2_known_port(514).
c2_known_port(1022).
c2_port(P) :- c2_known_port(P).

% DARPA attack phase rules

icmp_req(SO, DO, Proto) :-
    ext_orig(SO),
    loc_resp(DO),
    icmp(Proto).

icmp_resp(SO, DO, Proto) :-
    loc_orig(SO),
    ext_resp(DO),
    icmp(Proto).

sadmind_req(SO, DO, DPort, Proto) :-
    ext_orig(SO),
    loc_resp(DO),
    sadmind_port(DPort),
    udp(Proto).

sadmind_exp(SO, DO, DPort, Proto) :-
    ext_orig(SO),
    loc_resp(DO),
    sadmind_followup_port(DPort),
    tcp(Proto).

c2_direction(SO, DO, DPort, Proto) :-
    loc_orig(SO),
    ext_orig(DO),
    c2_port(DPort),
    tcp(Proto).

suspicious_download(SO, DO, DPort, Proto) :-
    ext_orig(SO),
    loc_resp(DO),
    download_port(DPort),
    tcp(Proto).


% Multi-step attack logic

phase(1, X, SO, DO, _, Proto, _, _, phase1) :-
    icmp_req(SO, DO, Proto),
    recon(X, phase1).

phase(1, X, SO, DO, _, Proto, _, _, benign) :- 
    \+ phase(1, X, SO, DO, _, Proto, _, _, phase1).


phase(2, X, SO, DO, DPort, Proto, _, _, phase2) :-
    (sadmind_req(SO, DO, DPort, Proto); icmp_resp(SO, DO, Proto)),
    ping(X, phase2).

phase(2, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(2, X, SO, DO, DPort, Proto, _, _, phase2).
    

phase(3, X, SO, DO, DPort, _, _, _, phase3) :- 
    (sadmind_req(SO, DO, DPort, Proto); sadmind_exp(SO, DO, DPort, Proto)),
    overflow(X, phase3).

phase(3, X, SO, DO, DPort, _, _, _, benign) :-
    \+ phase(3, X, SO, DO, DPort, _, _, _, phase3).


phase(4, X, SO, DO, DPort, Proto, _, _, phase4) :-
    (c2_direction(SO, DO, DPort, Proto); suspicious_download(SO, DO, DPort, Proto)),
    install(X, phase4).

phase(4, X, SO, DO, DPort, Proto, _, _, benign) :-
    \+ phase(4, X, SO, DO, DPort, Proto, _, _, phase4).


0.9 :: traffic_spike(R) :- R > 1.
0.5 :: traffic_spike(R) :- R > 0.5, R < 1.

phase(5, X, _, _, _, _, R, S, phase5) :-
    traffic_spike(R),
    ddos(X, phase5).

phase(5, X, _, _, _, _, R, S, benign) :-
    \+ phase(5, X, _, _, _, _, R, S, phase5).


multi_step(X, Next, SO, DO, DPort, Proto, R, S, Outcome) :-
    phase(Next, X, SO, DO, DPort, Proto, R, S, Outcome). 