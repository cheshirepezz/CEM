%close all
clear all

x0=[0,0,0];
v0=[1e3,0,0];
E0=[0,0,0e-4]*0;
B0=[0,0,1e-4];
B1=[1e-4,0,0];

e=1.602176e-19;
m=9.109e-31;
qom=e/m;

wc=qom*norm(B0);



Time=5*2*pi/wc;
dt=.1/wc;

NT=round(Time/dt);

v=v0-.5*dt*qom*(E0+cross(v0,B0));
x=x0;
for it=1:NT
    x=x+dt*v;
    E=E0;
    B=B0+B1*x(2);
%
% Boris mover
%
qomdt2=dt*qom/2;
h=qomdt2*B;
s=2*h/(1+dot(h,h));
u=v+qomdt2*E;
up=u+cross(u+cross(u,h),s);
v=up+qomdt2*E;
%
% Bad Mover
%
%v=v+dt*qom*(E+cross(v,B));

color='b';

subplot(2,2,1)    
plot(x(1),v(1),color)
xlabel('x(1)')
ylabel('v(1)')
hold on
subplot(2,2,2)    
plot(x(2),v(2),color)
xlabel('x(2)')
ylabel('v(2)')
hold on
subplot(2,2,3)    
plot(x(3),v(3),color)
xlabel('x(3)')
ylabel('v(3)')
hold on
subplot(2,2,4)    
plot(x(1),x(2),color)
xlabel('x(1)')
ylabel('x(2)')
hold on
end    