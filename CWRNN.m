
classdef CWRNN < handle
   properties
       
       h_dim %Hidden state dimension
       freq_number
       n_i
       h_tot
       
       working_freq
       
       input_set
       desired_set
       
       srnn_array=[]
       
       W_d
       d_t
       
       final_outputs
       
       update
       update_pre
   end
   
   
   methods
        function obj=CWRNN(input_set,desired_set,h_dim,freq_number)
            
            %Initialize properties.
            obj.input_set=input_set;
            obj.desired_set=desired_set;
            
            obj.final_outputs=zeros(1,size(input_set,2));
            
            obj.n_i=size(input_set,1)+1;
            obj.h_dim=h_dim;
            
            %Initialize weights.
            obj.W_d=(0.1)*randn(1,freq_number*h_dim+obj.n_i);
            
            %Initialize state vector as a zero vector.
            obj.h_tot=zeros(freq_number*h_dim,1);
            obj.d_t=0;
            
            %Initialize working frequencies.
            obj.freq_number=freq_number;
            obj.working_freq=2.^(0:freq_number-1);
            
            %Initialize inner RNN.
            for i=1:freq_number
                obj.srnn_array=[obj.srnn_array,SRNN_aux((i-1)*obj.h_dim+obj.n_i,obj.h_dim)];
            end
            obj.srnn_array=fliplr(obj.srnn_array);
            
            obj.update=zeros(freq_number*h_dim+obj.n_i,1);
            obj.update_pre=zeros(freq_number*h_dim+obj.n_i,1);
            
        end
        
        function soft_reset(obj)
            
     
            %Make state vectors and error propagation matrices zero.
            %Do not change the weights of the RNN.
            obj.final_outputs=zeros(1,size(obj.input_set,2));
            
            obj.h_tot=zeros(obj.freq_number*obj.h_dim,1);
            for i=1:obj.freq_number
                obj.srnn_array(i).soft_reset();
                
            end

        end
        
        
        
        function reset(obj)
            
            %Make state vectors and error propagation matrices zero.
            %DChange the weights of the RNN.
            obj.W_d=sqrt(0.1)*randn(1,obj.freq_number*obj.h_dim+obj.n_i);
     
            
            obj.final_outputs=zeros(1,size(obj.input_set,2));
            
            obj.h_tot=zeros(obj.freq_number*obj.h_dim,1);
            for i=1:obj.freq_number
                obj.srnn_array(i).hard_reset();
                
            end
            
            obj.update=zeros(obj.freq_number*obj.h_dim+obj.n_i,1);
            obj.update_pre=zeros(obj.freq_number*obj.h_dim+obj.n_i,1);
        end
        
        function error_vec=forward(obj,input_index,input_inst)
            
            
            %Forward CW-RNN operation.
            h_int=zeros(obj.freq_number*obj.h_dim,1);
            forward_vector=zeros( (obj.freq_number-1)*obj.h_dim+obj.n_i,1);
            forward_vector(end-obj.n_i+1:end)=input_inst;
            for k=obj.freq_number:-1:1
                
                if(k==obj.freq_number && k>1)
                    if(mod(input_index,obj.working_freq(k))==0)
                        obj.srnn_array(k).forward(input_inst);
                    end
                    forward_vector((k-2)*obj.h_dim+1:(k-1)*obj.h_dim)=obj.srnn_array(k).h_t;
                elseif(k<obj.freq_number && k>1)
                    if(mod(input_index,obj.working_freq(k))==0)
                        obj.srnn_array(k).forward(forward_vector((k-1)*obj.h_dim+1:end));
                    end
                    forward_vector((k-2)*obj.h_dim+1:(k-1)*obj.h_dim)=obj.srnn_array(k).h_t;
                elseif(k==1)
                    obj.srnn_array(k).forward(forward_vector); %Freq i zaten 1.
                end

               h_int((k-1)*obj.h_dim+1:k*obj.h_dim)=obj.srnn_array(k).h_t;
               
            end
            obj.h_tot=h_int;
            concat=[obj.h_tot;input_inst];
            obj.d_t=(obj.W_d*concat);
            error_vec=obj.desired_set(input_index)-obj.d_t;
        end
        
        function update_W_d(obj,error_vec,input_inst,lr,mom)
          
           %Since the last layer linear, Neterov update is equavlent to the
           %standard momentum update.
           dtWd=[obj.h_tot', input_inst'];
           obj.update=lr*dtWd'*error_vec;
           
                 
           update_mom=obj.update+mom*obj.update_pre;
           obj.update=update_mom;

           
           
           obj.W_d(:)=obj.W_d(:)+update_mom; 
           
           obj.update_pre=obj.update;
           
            

            
        end
        
        function evaluate(obj,x_input)
            
            %Evaluate CW-RNN weights by simulating whole process.
            total_time=size(obj.input_set,2);
            obj.final_outputs=zeros(1,total_time);
            obj.h_tot=zeros(obj.h_dim,1);
            for j=1:total_time
                if(mod(j,total_time)~=0)
                    i=mod(j,total_time);
                else
                    i=total_time;
                end

                obj.forward(i,x_input(:,i));
                obj.final_outputs(j)=obj.d_t;
            end
            figure(10); 
            plot(obj.desired_set,'LineWidth',1.5);grid
            hold on;
            plot(obj.final_outputs,'LineWidth',1.5);
            
            title([' Total Error ', num2str(sum(abs(obj.desired_set-obj.final_outputs)))]);
            hold off;
        end
        
        function SGD_train(obj,lr,mom,repeat)
            
            
            %Augment the input data
            total_time=size(obj.input_set,2);
            x_input=[obj.input_set; ones(1,total_time)]; 
            errors=zeros(1,repeat*total_time);
            
            for j=1:repeat*total_time
                if(mod(j,total_time)~=0)
                    i=mod(j,total_time);
                else
                    i=total_time;
                end
                
                %Foward CW-RNN and calculate error.
                error_vec=obj.forward(i,x_input(:,i));
                error=sqrt(error_vec'*error_vec);
                errors(j)=error;
                obj.final_outputs(i)=obj.d_t;
                
                %Update the weights.
                for k=1:obj.freq_number
                   if(mod(i,obj.working_freq(k))==0)
                       obj.srnn_array(k).error_propogate();
                   end
                   dthtm=zeros(1,obj.h_dim);
                   for l=1:k
                       
                       temp=obj.W_d((l-1)*obj.h_dim+1:l*obj.h_dim)*obj.d_t*(1-obj.d_t);
                       
                       if(l<k)
                           W_t_l=obj.srnn_array(l).W_h(:,(k-l-1)*obj.h_dim+1:(k-l)*obj.h_dim);
                           h_t_l=obj.srnn_array(l).h_t;
                           temp_mult=W_t_l.*(1-h_t_l.^2);
                           temp=temp*temp_mult;
                       end
                       
                       
                       dthtm=dthtm+temp;
                   end
                   
                   obj.srnn_array(k).update_weights(dthtm,error_vec,lr,mom);
                end
                
                obj.update_W_d(error_vec,x_input(:,i),lr,mom);
                
                fprintf('Iteration: %d, Error: %f \n',j,errors(j));

                if(i==total_time)
                   
                    figure(1); 
                    plot(obj.desired_set,'LineWidth',1.5);grid
                    hold on;
                    plot(obj.final_outputs,'LineWidth',1.5);
                    
                    title(['Epoch ' num2str(round(j/total_time)), ' Total Error ', num2str(sum(abs(obj.desired_set-obj.final_outputs)))]);
                    hold off;
                    
                    if(sum(abs(obj.desired_set-obj.final_outputs))<1e-3)
                        break;
                    else
                        obj.final_outputs=zeros(1,size(obj.input_set,2));
                    end
                    
                    obj.soft_reset();
                    obj.evaluate(x_input);
                    obj.soft_reset();
                end
                
                
            end
            
            fprintf('Training has ended. ## \n');
            figure(99); hold on; 
            plot(movmean(errors,500));

        end

   end
        

   
end