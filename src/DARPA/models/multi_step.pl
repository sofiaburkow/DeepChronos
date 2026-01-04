nn(phase_1_net, [X], Y, [0,1]) :: s1(X,Y).

query_pred(X,Y) :- s1(X,Y).