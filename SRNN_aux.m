
classdef SRNN_aux < handle
    %Auxiliary SRNN class for CW-RNN.
   properties
       
       h_dim %Hidden state dimension
       n_i % Augmented input vector dimension
       n_w % Number of weights in W_h
       input_vec
       
       h_t %hidden state at t
       h_t_1 %hidden state at t-1
       d_t
       
       
       W_h %Hidden state vectors
       %W_c_d %Corresponding output weights
       C_t %Error propogation- Matrix
       H_t
       
       update
       update_pre
 
   end
   methods
        function obj=SRNN_aux(input_dim,h_dim)
            
            
            obj.n_i=input_dim; 
            obj.h_dim=h_dim;
            
            obj.input_vec=zeros(input_dim,1);
            
            obj.n_w=obj.h_dim*(obj.h_dim+obj.n_i);
            obj.h_t=zeros(obj.h_dim,1);
            obj.h_t_1=zeros(obj.h_dim,1);
            obj.d_t=0;

            obj.W_h=(0.1)*randn(obj.h_dim,obj.h_dim+obj.n_i);
            
            
            obj.C_t=zeros(obj.h_dim,obj.n_w);
            obj.H_t=zeros(1,obj.n_w);
            obj.update=zeros(obj.n_w,1);
            obj.update_pre=zeros(obj.n_w,1);

        end
        
        function soft_reset(obj)
            obj.input_vec=zeros(obj.n_i,1);
            obj.h_t=zeros(obj.h_dim,1);
            obj.h_t_1=zeros(obj.h_dim,1);
            obj.d_t=0;


            obj.C_t=zeros(obj.h_dim,obj.n_w);
            obj.H_t=zeros(1,obj.n_w);
            
        end
        
        
        function hard_reset(obj)
            
            obj.input_vec=zeros(obj.n_i,1);
            obj.h_t=zeros(obj.h_dim,1);
            obj.h_t_1=zeros(obj.h_dim,1);
            obj.d_t=0;

            obj.W_h=(0.1)*randn(obj.h_dim,obj.h_dim+obj.n_i);
            
            
            obj.C_t=zeros(obj.h_dim,obj.n_w);
            obj.H_t=zeros(1,obj.n_w);
            obj.update=zeros(obj.n_w,1);
            obj.update_pre=zeros(obj.n_w,1);
            
        end
        
        
        function forward(obj,input_inst)
            obj.input_vec=input_inst;
            
            obj.h_t_1=obj.h_t;
            concat=[obj.input_vec; obj.h_t_1];
            obj.h_t=tanh(obj.W_h*concat);

            
        end
        
        function error_propogate(obj)
            obj.SGD_update(0,0.95,0); %Nesterov
            
            
            W_2_h=obj.W_h(:,obj.n_i+1:end);


             %Weightlere göre türevler.
            A= [ obj.input_vec' , obj.h_t_1' ];
            L_t=kron(eye(obj.h_dim),A);

            hthtm=W_2_h.*(1-obj.h_t.*obj.h_t);


            htWt=L_t.*(1-obj.h_t.*obj.h_t);

            obj.C_t=hthtm*obj.C_t+htWt;
            
        end
        
        function SGD_update(obj,lr,error_vec,mom)
           
            
           obj.update=lr*obj.H_t'*error_vec;
            
           update_mom=obj.update+mom*obj.update_pre;
           obj.update=update_mom;
           obj.W_h(:)=obj.W_h(:)+update_mom(1:obj.n_w);

        end
        
        function update_weights(obj,dthtm,error_vec,lr,mom)
            
            obj.H_t=dthtm*obj.C_t;
            
            obj.SGD_update(lr,error_vec,mom);
            obj.update_pre=obj.update;    
       

        end
        
   end
   
end