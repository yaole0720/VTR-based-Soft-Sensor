%% 二进制DE
function [Pb,Xb,x,trace]=IDE(D,DL,NP,gen_max,X,Y,w,tau)
% D：参数维度
% NP：种群规模
% pi：初始化群体的先验概率
% gen_max：最大迭代次数

CR=0.1;%交叉概率
eps=1e-9;%精度
F=0.5;

trace=zeros(gen_max,2);
rand('state',1);   
x=zeros(NP,D); %初始种群
for i=1:NP
    for j=1:D
        x(i,j)=floor(rand()*DL);
    end
end

count=1;%%count为当前代数
trial=zeros(1,D);
cost=zeros(1,NP);
cost(1)=fitness(x(1,:),X,Y,DL,w,tau);%% 适应度函数
Pb=cost(1);%存放最优值
Xb=x(1,:);%存放最优位置
for i=2:NP
    cost(i)=fitness(x(i,:),X,Y,DL,w,tau);
    if(cost(i)<=Pb)
        Pb=cost(i);
        Xb=x(i,:);                
   end
end
trace(1,1)=1;
trace(1,2)=Pb;
while(count<gen_max) %count<gen_max abs(Pb)>eps
    count;
    for i=1:NP
        i;
        while 2>1
            a=floor(rand*NP)+1;
            if a~=i
                break;
            end
        end
        while 2>1
            b=floor(rand*NP)+1;
            if b~=i&b~=a
                break;
            end
        end
        while 2>1
            c=floor(rand*NP)+1;
            if c~=i&c~=a&c~=b
                break;
            end
        end
        jrand=floor(rand*D+1);
        for k=1:D
            if(rand<CR|jrand==k)               
                trial_1(k)=floor(x(c,k)+F*(x(a,k)-x(b,k)));
                if (trial_1(k)>DL)
                    trial_1(k) = DL;
                elseif (trial_1(k)<0)
                    trial_1(k) = 0;
                end
                trial_2(k)=ceil(x(c,k)+F*(x(a,k)-x(b,k)));
                if (trial_2(k)>DL)
                    trial_2(k) = DL;
                elseif (trial_2(k)<0)
                    trial_2(k) = 0;
                end
            else
                trial_1(k)=x(i,k);
                trial_2(k)=x(i,k);
            end
        end
        score_1=fitness(trial_1,X,Y,DL,w,tau);
        score_2=fitness(trial_2,X,Y,DL,w,tau);
        if score_1<score_2
            trial = trial_1;
        else
            trial = trial_2;
        end
        
        score=fitness(trial,X,Y,DL,w,tau);
           if(score<=cost(i))
              x(i,1:D)=trial(1:D);
              cost(i)=score;
           end
           if cost(i)<=Pb
              Pb=cost(i);
              Xb(1:D)=x(i,1:D);
           end
    end  
    count=count+1;
    trace(count,1)=count;
    trace(count,2)=Pb;
end
%--------------结束搜索---------------
count;
Pb;
Xb;

