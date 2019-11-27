
classdef LSTM < handle
   properties
       
       h_dim %Hidden state dimension
       input_set
       desired_set
   end
   properties
       n_i % Augmented input vector dimension
       n_w % Number of weights in W_h
       
       n_theta % Total number of weights (i.e W_h and W_d)
       y_t %hidden state at t
       y_t_1 %hidden state at t-1
       c_t
       c_t_1
       d_t
       
       z_t
       i_t
       f_t
       o_t
       
       W_z
       W_i
       W_f
       W_o
       W_d
       H_t
       C_t
       D_t

       update
       update_pre
       
       del_ct
       final_outputs
   end
   methods
        function obj=LSTM(input_set,desired_set,h_dim)
            
            %Initialize properties.
            obj.input_set=input_set;
            obj.desired_set=desired_set;
            
            
            obj.n_i=size(input_set,1)+1; %All inputs are column vectors.
            obj.h_dim=h_dim;
   
            obj.n_w=obj.h_dim*(obj.h_dim+obj.n_i);
            obj.n_theta=(4*obj.h_dim+1)*(obj.h_dim+obj.n_i);
            
            %Initialize state vectors as zero vectors.
            obj.y_t=zeros(obj.h_dim,1);
            obj.y_t_1=zeros(obj.h_dim,1);
            obj.c_t=zeros(obj.h_dim,1);
            obj.c_t_1=zeros(obj.h_dim,1);
            obj.z_t=zeros(obj.h_dim,1);
            obj.i_t=zeros(obj.h_dim,1);
            obj.f_t=zeros(obj.h_dim,1);
            obj.o_t=zeros(obj.h_dim,1);
            
            obj.d_t=0;

            %Initialize weights.
            const=(0.1);
            obj.W_z=const*randn(obj.h_dim,obj.h_dim+obj.n_i);
            obj.W_i=const*randn(obj.h_dim,obj.h_dim+obj.n_i);
            obj.W_f=const*randn(obj.h_dim,obj.h_dim+obj.n_i);
            obj.W_f(:,obj.n_i)=5;
            obj.W_o=const*randn(obj.h_dim,obj.h_dim+obj.n_i);
            obj.W_d=const*randn(1,obj.h_dim+obj.n_i);
            
            %Initialize error propagation matrices..
            obj.del_ct=zeros(obj.h_dim,1);
            obj.H_t=zeros(1,obj.n_theta);
            
            obj.final_outputs=zeros(1,size(obj.input_set,2));
            obj.update=zeros(obj.n_theta,1);
            obj.update_pre=zeros(obj.n_theta,1);
        end
        
        function error_vec=forward(obj,input_inst,desired_inst)
            
            %Forward LSTM operation.
            
            obj.y_t_1=obj.y_t;
            concat=[input_inst; obj.y_t];
            obj.z_t=tanh(obj.W_z*concat); %Hidden state
            obj.i_t=sigmoid(obj.W_i*concat); %Hidden state filter
            obj.f_t=sigmoid(obj.W_f*concat); %Forget
            obj.c_t_1=obj.c_t;  % Cell-state- Long term info
            obj.c_t=obj.i_t.*obj.z_t + obj.f_t .* obj.c_t_1; %Update cell state
            obj.o_t=sigmoid(obj.W_o*concat); % Output-filter
            obj.y_t=obj.o_t.*tanh(obj.c_t); %Output-info
            obj.d_t=(obj.W_d*[input_inst; obj.y_t]); %Data estimation


            error_vec=desired_inst-obj.d_t;
        end
        
        function backward(obj,input_inst)
            
            %Backpropagation (or RTRL) operation.
            W_2_d=obj.W_d(:,obj.n_i+1:end); %Vectorize output layer weights.
            dtyt=W_2_d'; % del d_t / del y_t
            A= [ input_inst' , obj.y_t' ];
            dttt=A;

            del_ot=dtyt.*tanh(obj.c_t);
            obj.del_ct=obj.del_ct+dtyt.*obj.o_t.*(1-tanh(obj.c_t).^2);
            del_it=obj.del_ct.*obj.z_t;
            del_ft=obj.del_ct.*obj.c_t_1;
            del_zt=obj.del_ct.*obj.i_t;


            del_zth=del_zt.*(1-obj.z_t.^2);
            del_ith=del_it.*obj.i_t.*(1-obj.i_t);
            del_fth=del_ft.*obj.f_t.*(1-obj.f_t);
            del_oth=del_ot.*obj.o_t.*(1-obj.o_t);

            del_ath=[del_zth; del_ith; del_fth; del_oth];
            del_W=del_ath*[ input_inst ; obj.y_t_1 ]';
            del_W=del_W(:);
            obj.H_t=[del_W',dttt];
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
            obj.W_z(:)=obj.W_z(:)+update_mom(1:obj.n_w);
            obj.W_i(:)=obj.W_i(:)+update_mom(obj.n_w+1:2*obj.n_w);
            obj.W_f(:)=obj.W_f(:)+update_mom(2*obj.n_w+1:3*obj.n_w);
            obj.W_o(:)=obj.W_o(:)+update_mom(3*obj.n_w+1:4*obj.n_w);
            obj.W_d(:)=obj.W_d(:)+update_mom(4*obj.n_w+1:end);
            
            
            
        end
        
        function evaluate(obj,x_input)
            
            %Evaluate LSTM weights by simulating whole process.
            total_time=size(obj.input_set,2);
            ev_final_outputs=zeros(1,total_time);
            ev_c_t=zeros(obj.h_dim,1);
            ev_y_t=zeros(obj.h_dim,1);
            for j=1:total_time
                if(mod(j,total_time)~=0)
                    i=mod(j,total_time);
                else
                    i=total_time;
                end

                %Forward Equation.
                ev_y_t_1=ev_y_t;
                concat=[x_input(:,i); ev_y_t_1];
                ev_z_t=tanh(obj.W_z*concat); %Hidden state
                ev_i_t=sigmoid(obj.W_i*concat); %Hidden state filter
                ev_f_t=sigmoid(obj.W_f*concat); %Forget
                ev_c_t_1=ev_c_t;  % Cell-state- Long term info
                ev_c_t=ev_i_t.*ev_z_t + ev_f_t .* ev_c_t_1; %Update cell state
                ev_o_t=sigmoid(obj.W_o*concat); % Output-filter
                ev_y_t=ev_o_t.*tanh(ev_c_t); %Output-info
                ev_d_t=(obj.W_d*[x_input(:,i); ev_y_t]); %Data estimation
                ev_final_outputs(j)=ev_d_t;
            end
            figure(10); 
            plot(obj.desired_set,'LineWidth',1.5);grid
            hold on;
            plot(ev_final_outputs,'LineWidth',1.5);
            
            obj.final_outputs=ev_final_outputs;
            
            title([' Total Error ', num2str(sum(abs(obj.desired_set-ev_final_outputs)))]);
            hold off;
            
        end
        
        
        
        function SGD_train(obj,lr,mom,repeat)
            
            %Train LSTM weights.
            total_time=size(obj.input_set,2);
            x_input=[obj.input_set; ones(1,total_time)]; 
            errors=zeros(1,repeat*total_time);
            
            pre_error_sum=0;
           for j=1:repeat*total_time
                if(mod(j,total_time)~=0)
                    i=mod(j,total_time);
                else
                    i=total_time;
                end
                
                %Forward LSTM and calculate error vector.
                %Backward opration- Calculate gradients.
                error_vec=obj.forward_backward(x_input(:,i),obj.desired_set(i));
                obj.final_outputs(i)=obj.d_t;
                
                error=error_vec'*error_vec;
                errors(j)=error;

                fprintf('Iteration: %d, Error: %f \n',j,sqrt(error'*error));

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
                        
                        %Soft Reset at the beginning of the new epoch.
                        obj.final_outputs=zeros(1,size(obj.input_set,2));
                        obj.C_t=zeros(obj.h_dim,obj.n_theta);
                        obj.D_t=zeros(obj.h_dim,obj.n_theta);
                        obj.H_t=zeros(1,obj.n_theta);

                        obj.y_t=zeros(obj.h_dim,1);
                        obj.y_t_1=zeros(obj.h_dim,1);
                        obj.c_t=zeros(obj.h_dim,1);
                        obj.c_t_1=zeros(obj.h_dim,1);
                        obj.z_t=zeros(obj.h_dim,1);
                        obj.i_t=zeros(obj.h_dim,1);
                        obj.f_t=zeros(obj.h_dim,1);
                        obj.o_t=zeros(obj.h_dim,1);

                        obj.d_t=0;

                
                    end
                    
                    
                    

                end


           end 
           
           %Plot errors.
           fprintf('Training has ended. Error : %f ## \n',pre_error_sum);
           figure; 
           plot(1:j,movmean(errors(1:j),25));
            
           
            
        end
        

       
   end
    
end