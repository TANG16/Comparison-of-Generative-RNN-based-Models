
classdef SRNN < handle
   properties
       
       h_dim %Hidden state dimension
       input_set
       desired_set
   end
   properties
       n_i % Augmented input vector dimension
       n_w % Number of weights in W_h
       
       n_theta % Total number of weights (i.e W_h and W_d)
       h_t %hidden state at t
       h_t_1 %hidden state at t-1
       d_t
       
       
       W_h
       W_d
       H_t
       C_t
        

       
       update
       update_pre
       
       final_outputs
   end
   methods
        function obj=SRNN(input_set,desired_set,h_dim)
            
            %Initialize properties.
            obj.input_set=input_set;
            obj.desired_set=desired_set;
            
            
            
            obj.n_i=size(input_set,1); %All inputs are column vectors.
            obj.h_dim=h_dim;
            obj.n_w=obj.h_dim*(obj.h_dim+obj.n_i);
            obj.n_theta=(obj.h_dim+1)*(obj.h_dim+obj.n_i);
            
            %Initialize state vectors as zero vectors.
            obj.h_t=zeros(obj.h_dim,1);
            obj.h_t_1=zeros(obj.h_dim,1);
            obj.d_t=0;

            %Initialize weights.
            obj.W_h=(0.1)*randn(obj.h_dim,obj.h_dim+obj.n_i);
            obj.W_d=(0.1)*randn(1,obj.h_dim+obj.n_i);
            obj.H_t=zeros(1,obj.n_theta);
            obj.C_t=zeros(obj.h_dim,obj.n_theta);
            

            obj.final_outputs=zeros(1,size(obj.input_set,2));
            
            obj.update=zeros(obj.n_theta,1);
            obj.update_pre=obj.update;
        end
        
        function soft_reset(obj)
            %Make state vectors and error propagation matrices zero.
            %Do not change the weights of the RNN.
            obj.d_t=0;

            obj.H_t=zeros(1,obj.n_theta);
            obj.C_t=zeros(obj.h_dim,obj.n_theta); 
            obj.h_t=zeros(obj.h_dim,1);
            obj.h_t_1=zeros(obj.h_dim,1);
            
            obj.final_outputs=zeros(1,size(obj.input_set,2));
        end
        
        function hard_reset(obj)
            %Make state vectors and error propagation matrices zero.
            %Change the weights of the RNN.
            
            obj.d_t=0;

            obj.W_h=(0.1)*randn(obj.h_dim,obj.h_dim+obj.n_i);
            obj.W_d=(0.1)*randn(1,obj.h_dim+obj.n_i);
            obj.H_t=zeros(1,obj.n_theta);
            obj.C_t=zeros(obj.h_dim,obj.n_theta); 
            obj.h_t=zeros(obj.h_dim,1);
            obj.h_t_1=zeros(obj.h_dim,1);
            

            obj.update=zeros(obj.n_theta,1);
            obj.update_pre=obj.update;

            
            obj.final_outputs=zeros(1,size(obj.input_set,2));
        end
        
        function error_vec=forward(obj,input_inst,desired_inst)
            
            %Foward the RNN.
            obj.h_t_1=obj.h_t;
            concat=[input_inst; obj.h_t_1];
            obj.h_t=tanh(obj.W_h*concat);
            obj.d_t=(obj.W_d*[input_inst; obj.h_t]); %Data estimation
            
            error_vec=desired_inst-obj.d_t;
        end
        
        function backward(obj,input_inst)
            
            %Backpropagation (or RTRL) operation.
            W_2_h=obj.W_h(:,obj.n_i+1:end);
            W_2_d=obj.W_d(:,obj.n_i+1:end);

            %Derivatives wrt weights.
            A= [ input_inst' , obj.h_t_1' ];
            L_t=kron(eye(obj.h_dim),A);
            
            
            %Derivatives of h_t
            hthtm=W_2_h.*(1-obj.h_t.*obj.h_t);
            dtht=W_2_d;
            htw=L_t.*(1-obj.h_t.*obj.h_t);
            httt=[htw, zeros(obj.h_dim,obj.h_dim+obj.n_i)];
            
            %Derivatives of d_t
            A= [input_inst' , obj.h_t'];
            dttt=A;

            %Error propagate.
            obj.C_t=hthtm*obj.C_t+httt;
            obj.H_t=dtht*obj.C_t+[zeros(1,obj.n_w), dttt];
            
            if(max(abs(obj.H_t))>1e5)
                %Throw error if the gradient explode.
                msg = 'Exploding Gradient!';
                error(msg)
            end
        end
        
        function evaluate(obj,x_input)
            
            %Evaluate RNN weights by simulating whole process.
            total_time=size(obj.input_set,2);
            final_ev_outputs=zeros(1,total_time);
            ev_h_t=zeros(obj.h_dim,1);
            for j=1:total_time
                if(mod(j,total_time)~=0)
                    i=mod(j,total_time);
                else
                    i=total_time;
                end

                %Forward Equation.
                concat=[x_input(:,i); ev_h_t];
                ev_h_t=tanh(obj.W_h*concat);
                ev_d_t=(obj.W_d*[x_input(:,i); ev_h_t]);

                final_ev_outputs(j)=ev_d_t;
            end
            figure(10); 
            plot(obj.desired_set,'LineWidth',1.5);grid
            hold on;
            plot(final_ev_outputs,'LineWidth',1.5);
            
            title([' Total Error ', num2str(sum(abs(obj.desired_set-final_ev_outputs)))]);
            hold off;
            
        end
        
        function error_vec=forward_backward(obj,input_inst,desired_inst)
            
            %Calculate look-ahead gradient term- Nesterov AGD.
            obj.SGD_update(0,0.95,0); 
            
            error_vec=obj.forward(input_inst,desired_inst);
                
            obj.backward(input_inst)
            
        end
        

        function SGD_update(obj,lr,mom,error_vec)
            
           %Update weights with SGD.
            
           obj.update=lr*obj.H_t'*error_vec;
            
           update_mom=obj.update+mom*obj.update_pre;
           obj.update=update_mom; 
           
           obj.W_h(:)=obj.W_h(:)+update_mom(1:obj.n_w);
           obj.W_d(:)=obj.W_d(:)+update_mom(obj.n_w+1:end); 
           
           
           
        end
        
        function SGD_train(obj,lr,mom,repeat)
            
            total_time=size(obj.input_set,2);
            x_input=[obj.input_set]; 
            errors=zeros(1,repeat*total_time);
            
            %Plotting.
%             h=figure(1);
%             figure(h);
%             plot(obj.desired_set,'LineWidth',1.5);grid
%             hold on;
%             lnh=plot(obj.final_outputs);
%             lnh.YDataSource='obj.final_outputs';
%             hold off;
            
            %Train RNN weights.
            for j=1:repeat*total_time
                if(mod(j,total_time)~=0)
                    i=mod(j,total_time);
                else
                    i=total_time;
                end
                
               
                %Forward RNN and calculate error vector.
                %Backward opration- Calculate gradients.
                error_vec=obj.forward_backward(x_input(:,i),obj.desired_set(i));
                obj.final_outputs(i)=obj.d_t;
                error=sqrt(error_vec'*error_vec);
                errors(j)=error;
                
                
                fprintf('Iteration: %d, Error: %f \n',j,errors(j));
                
                
                %Update Weights.
                obj.SGD_update(lr,mom,error_vec);
                obj.update_pre=obj.update;
                
                
                
           
                
                if(i==total_time)
                   
                    figure(1); 
                    plot(obj.desired_set,'LineWidth',1.5);grid
                    hold on;
                    plot(obj.final_outputs,'LineWidth',1.5);
                    
                    title(['Epoch ' num2str(round(j/total_time)), ' Total Error ', num2str(sum(abs(obj.desired_set-obj.final_outputs)))]);
                    hold off;
                    hold off;
                    obj.evaluate(x_input);
                    if(sum(abs(obj.desired_set-obj.final_outputs))<5)
                        break;
                    else
                        obj.final_outputs=zeros(1,size(obj.input_set,2));
                        obj.H_t=zeros(1,obj.n_theta);
                        obj.C_t=zeros(obj.h_dim,obj.n_theta); 
                        obj.h_t=zeros(obj.h_dim,1);
                        obj.h_t_1=zeros(obj.h_dim,1);
                        obj.evaluate(x_input);
                    end
                end
                
            end
            fprintf('Training has ended. Total Error: %f \n',sum(abs(obj.desired_set-obj.final_outputs)));
            figure(99); hold on; 
            plot(movmean(errors,500));
            
            %obj.evaluate(x_input);
            

        end
        

        
        
       
   end
    
end