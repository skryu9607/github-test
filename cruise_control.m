clear all;close all;clc

sysCar = ss(-1,1,1,0);

K = 5;
sysPROP = ss(-1-K,K,1,0);
%%
t = 0:0.01:10;
[yCar,tCar,xCar] = step(sysCar,t);
plot(tCar,yCar)
hold on
[yprop,tprop,xprop] = step(sysPROP,t);
plot(tprop,yprop)

% Need Integral action to reduce steady state error
A = [-1-K K;-1 0];
B = [K;1];
C = [1 0];
D = 0;
sys_integral = ss(A,B,C,D);
[ypi,tpi,xpi] = step(sys_integral,t);
plot(tpi,ypi)