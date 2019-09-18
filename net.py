import torch.nn as nn


class Net(nn.Module):
    '''
    Neural Network Class
    net_layer: list with the number of neurons for each network layer, [n_imput, ..., n_output]
    '''
    def __init__(self, 
                 layers_size= [2, 2], 
                 out_size = 1,
                 act= 'sig',
                 params_list= None, 
                 save_int= False):
        
        super(Net, self).__init__()
        
        self.layers = nn.ModuleList()
        
        if act == 'sig':
            self.activation = nn.Sigmoid()
        if act == 'tanh':
            self.activation = nn.Tanh()
        if act == 'relu':
            self.activation = nn.ReLU()

        # Save or not internal representations            
        self.save_int = save_int
        
        if save_int:
            self.int_rep = []
            
                   
        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k+1]))
            
        # Output layer
        self.out = nn.Linear(layers_size[-1], out_size)
        
        m_aux = 0
        
        for m in self.layers:
            
            if params_list is None:
                # Weights and biases initialization: uniform (-1, 1)
                nn.init.uniform_(m.weight, a= -1.0, b= 1.0)
                nn.init.uniform_(m.bias, a= -1.0, b= 1.0)
                
            else:
                m.weight = params_list[m_aux]
                m.bias = params_list[m_aux + 1]
                m_aux += 1
                
        if params_list is None:
            # Weights and biases initialization: uniform (-1, 1)
            nn.init.uniform_(self.out.weight, a= -1.0, b= 1.0)
            nn.init.uniform_(self.out.bias, a= -1.0, b= 1.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]
      
        
        
    def forward(self, x):
                
        for layer in self.layers:
                       
            # Activation function
            x = self.activation(layer(x))
            
            # If save_int is set to True
            if self.save_int:
                self.int_rep.append(x)

            
        # Last layer
        output = self.activation(self.out(x))
        
        # If save_int is set to True
        if self.save_int:
            self.int_rep.append(output)
        
        return output
    
    
    
    def internal_rep(self):
        
        if self.save_int:
            return self.int_rep
        else:
            return print('save_int=False')