f = 0.5; %定义频率；
w = 2*pi*f; %定义角频率；
nMax = 400; %定义采样点数；
ts = 1/10; %定义采样时间间隔；
n = 0:nMax; %采样序列；
x = exp(1i*w*n*ts);%复指数函数形式；

plot3(n*ts, real(x), imag(x));
t = n*ts;
close(mov);
mov = VideoWriter('exp', 'MPEG-4'); %定义生成动画文件名及格式；
mov.FrameRate = 20; %定义动画播放的帧率；
mov.Quality = 20; %定义视频动画的质量；

open(mov) %打开视频动画文件进行采集；

for ni = 0:nMax %开始循环，每次循环生成一帧图像
    ti = ni*ts; %定义循环体内的采样时间点；
    omega = 1i*w*ti; %角频率；
    xi = exp(omega); %复指数函数；
    clf %清除之前一帧的图像，这里是为了生成动态图，之前的图像清除后可以清楚看到指针转动并对应cos和sin的图像；
    % phasor
    subplot(2,2,[1,3]); % 生成一个2*2的图框；
    hold on %在绘制下一个图像时不清除上一个图像；
    plot([-1.1,1.1],[0,0],"k"); %绘制x轴线段；
    plot([0,0],[-1.1,1.1],'k'); %绘制y轴线段；
    
    axis equal %data edge equals to axis
    axis([-1.1,1.1,-1.1,1.1]) %set x from -1.1 to 1.1, set y from -1.1 to 1.1
    plot(x); %绘制x基底；
    plot(real(xi),imag(xi),'rd'); %绘制当前采样点；
    plot([0,real(xi)],[0,imag(xi)],'c'); %绘制原点到采样点的线段；
    xlabel('in-phase');%该图像x轴标签；
    ylabel('quadrature');%该图像y轴标签；
    title('phasor');%该图像标题；
    
    %以下内容与第一部分一致，不再重复。
    
    % cos
    subplot(2,2,2) 
    hold on
    plot([0,4],[0,0],'k');
    axis([0,4,-1.1,1.1]);
    plot(t, real(x));
    plot(ti, real(xi),'rd');
    plot([ti,ti], [0,real(xi)],'c-');
    xlabel('time (s)');
    ylabel('cos(wt)')
    title('cos');
    
    
    % sin
    subplot(2,2,4)
    hold on
    plot([0,4],[0,0],'k');
    axis([0,4,-1.1,1.1]);
    plot(t, imag(x));
    plot(ti, imag(xi),'rd');
    plot([ti,ti],[0,imag(xi)],'c-');
    xlabel('time (s)');
    ylabel('sin(wt)')
    title('sin');
    currentFrame = getframe(gcf);
    writeVideo(mov,currentFrame)

end
close(mov);