%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Phase Detectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nn(net1, [X], Z, [benign, attack]) :: recon_det(X, Z).
nn(net2, [X], Z, [benign, attack]) :: ping_det(X, Z).
nn(net3, [X], Z, [benign, attack]) :: overflow_det(X, Z).
nn(net4, [X], Z, [benign, attack]) :: install_det(X, Z).
nn(net5, [X], Z, [benign, attack]) :: ddos_det(X, Z).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Network topology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

homenet('172.16.115.20').
homenet('172.16.112.10').
homenet('172.16.112.50').

attacker('202.77.162.213').

exist_host(IP) :-
    homenet(IP).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Services
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sadmind_port(111).
sadmind_port(32773).
sadmind_port(32774).

open_port('172.16.115.20', 111).
open_port('172.16.112.10', 111).
open_port('172.16.112.50', 111).

sadmind_service(IP,Port) :-
    open_port(IP,Port),
    sadmind_port(Port).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Alert abstractions 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alert(recon,T, Attacker, Victim) :-
    recon_det(T,attack), 
    attacker(SIP),
    exist_host(DIP).

alert(ping,T, Attacker, Victim) :-
    ping_det(T,attack), 
    attacker(SIP),
    sadmind_service(DIP,_).

alert(exploit,T, Attacker, Victim) :-
    exploit_det(T,attack),
    attacker(SIP),
    sadmind_service(DIP,_).

alert(install,T, Attacker, Victim) :-
    install_det(T,attack),
    attacker(SIP),
    exist_host(DIP).

alert(ddos,T,SIP,Target) :-
    ddos_det(T,attack).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Attack graph edges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

edge(recon,ping).
edge(ping,exploit).
edge(exploit,install).
edge(install,ddos).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temporal ordering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

before(T1,T2) :-
    T1 < T2.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Attack chain reasoning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

chain(Type,T,SIP,DIP) :-
    alert(Type,T,SIP,DIP).

chain(Type2,T2,SIP,DIP) :-
    edge(Type1,Type2),
    chain(Type1,T1,SIP,DIP),
    alert(Type2,T2,SIP,DIP),
    before(T1,T2).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final attack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

multi_step_attack :-
    chain(ddos,_).