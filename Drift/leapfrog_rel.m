close all
clear all

e=1.602176e-19;
m=9.109e-31;
c= 299792458;

qom=e/m;

caso = 4
switch caso
case 1
%
% Case 1 - harmonic oscillator
%
E0=[0,0,0];
B0=[0,0,1e-7];
B1=[0,0,0];
E1=[-1e-3,0,0];
wcdt=.01; % Try 0.1 and 0.01
x0=[0,0,0];
v0=[c*.999,0,0]; % Try .999 and .99 and .9
gamma0=1/sqrt(1-dot(v0,v0)/c^2)
u0=v0.*gamma0;
wc=qom*norm(B0)/gamma0;

case 2
%
% Case 2 - relativistic gyromotion
%
E0=[0,0,0];
B0=[0,0,1e-4];
B1=[0,0,0];
E1=[0,0,0];
wcdt=.1; 
x0=[0,0,0];
v0=[c*.99,0,0];
gamma0=1/sqrt(1-dot(v0,v0)/c^2)
u0=v0.*gamma0;
%try correct relativisitc definition of wc and the classical
wc=qom*norm(B0)/gamma0;
%wc=qom*norm(B0);

case 3
%
% Case 3 - relativistic gyromotion with acceleration along B0
%
E0=[0,0,1e3];
B0=[0,0,1e-4];
B1=[0,0,0];
E1=[0,0,0];
wcdt=.1; 
x0=[0,0,0];
v0=[c*.9,0,0];
gamma0=1/sqrt(1-dot(v0,v0)/c^2)
u0=v0.*gamma0;
wc=qom*norm(B0)/gamma0;

case 4
%
% Case 4 - relativistic ExB, 
%
E0=[1e3,0,0];
B0=[0,0,1e-4];
B1=[0.0,0,0];
E1=[0,0,0];
wcdt=.1; 
x0=[0,0,0];
v0=[c*.99,0,0];
gamma0=1/sqrt(1-dot(v0,v0)/c^2)
u0=v0.*gamma0;
wc=qom*norm(B0)/gamma0;

otherwise
disp('Case not defined')
end






Time=15*2*pi/wc;
dt=wcdt/wc; 

NT=round(Time/dt);

u=u0-.5*dt*qom*(E0+cross(v0,B0));
x=x0;

xplt=[];
yplt=[];
zplt=[];
uxplt=[];
uyplt=[];
uzplt=[];

for it=1:NT
    gamma=sqrt(1+dot(u,u)/c^2);
    x=x+dt*u/gamma;
    E=E0+E1*x(1);
    B=B0+B1*x(2);
%
% Boris mover
%
qomdt2=dt*qom/2;
w=u+qomdt2*E;
gamman=sqrt(1+dot(w,w)/c^2);
h=qomdt2*B/gamman;
s=2*h/(1+dot(h,h));

up=w+cross(w+cross(w,h),s);
u=up+qomdt2*E;

xplt=[xplt;x(1)];
yplt=[yplt;x(2)];
zplt=[zplt;x(3)];
uxplt=[uxplt;u(1)];
uyplt=[uyplt;u(2)];
uzplt=[uzplt;u(3)];
end  

gammaplt=sqrt(1+(uxplt.^2+uyplt.^2+uzplt.^2)/c^2);
vxplt=uxplt./gammaplt;
vyplt=uyplt./gammaplt;
vzplt=uzplt./gammaplt;

color='b';

figure(1)
subplot(2,2,1)    
plot(xplt,uxplt,color)
xlabel('x(1)')
ylabel('u(1)')
title('x,ux')

subplot(2,2,2)    
plot(yplt,uyplt,color)
xlabel('x(2)')
ylabel('u(2)')
title('y,uy')

subplot(2,2,3)    
plot(zplt,uzplt,color)
xlabel('x(3)')
ylabel('u(3)')
title('z,uz')

subplot(2,2,4)    
plot(xplt,yplt,color)
xlabel('x(1)')
ylabel('x(2)')
title('x,y')

figure(2)
subplot(2,2,1)    
plot(xplt,vxplt,color)
xlabel('x(1)')
ylabel('v(1)')
title('x,vx')

subplot(2,2,2)    
plot(yplt,vyplt,color)
xlabel('x(2)')
ylabel('v(2)')
title('y,vy')

subplot(2,2,3)    
plot(zplt,vzplt,color)
xlabel('x(3)')
ylabel('v(3)')
title('z,vz')

figure(3)
subplot(2,2,1)    
plot(xplt)
xlabel('x(1)')
ylabel('v(1)')
title('x,vx')

subplot(2,2,2)    
plot(yplt)
xlabel('x(2)')
ylabel('v(2)')
title('y,vy')

subplot(2,2,3)    
plot(zplt)
xlabel('x(3)')
ylabel('v(3)')
title('z,vz')

subplot(2,2,4)    
plot(gammaplt)
xlabel('cycle')
ylabel('gammaplt')
title('cycle-gamma')