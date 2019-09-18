import torch
import torch.nn as nn
import torch.optim as optim
import time

X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])


y = torch.Tensor([[0], [1],
                  [1], [0]])

# I put one more layer. It learns with [2, 2, 2, 2, 2, 2, 2, 2]

net = Net(layers_size= [2, 2, 2, 2, 2, 2, 2, 2, 2],
          out_size = 1,
          act= 'tanh',
          save_int= False)

optimizer = optim.SGD(net.parameters(), 
                      lr= 0.1, 
                      weight_decay= 0.0)

#loss_func = nn.BCELoss()
#loss_func = nn.MSELoss()
loss_func = nn.L1Loss()


get_parameters= False
if get_parameters:
    params = []
    params.add(net.state_dict())
    

epochs= 5000000

t0_total = time.time()

for epoch in range(epochs):
    
    t0 = time.time()
             
    prediction= net(X)   
    loss = loss_func(prediction, y)     
          
    optimizer.zero_grad()   # Clear gradients for the next mini-batches
    loss.backward()         # Backpropagation, compute gradients
    optimizer.step()
    
    if get_parameters:
        params.add(net.state_dict())
    
    t1 = time.time()
    
    if epoch % 100 == 0:
        print('epoch: %d - loss: %.5f - time: %.4f' % (epoch, loss.item(), (t1-t0)))
        
t1_total = time.time()

time_elapsed = (t1_total - t0_total)/60.
          
    
print('prediction: ', net(X))
print('total time: %.2 min' % time_elapsed )
               
    ### Recording some results
    #log_dic['epoch'].append(epoch)
    #log_dic['loss_train'].append(np.mean(loss_epoch))
    #test_error_ep = loss_func(net(X_test), y_test).item()
    #log_dic['loss_test'].append(test_error_ep)
    #t1 = time.time()
    
    ### Training status
    #print('Epoch %d, Loss_train= %.10f, Loss_test= %.10f, Time= %.4f' % (epoch, 
    #                                                                     np.mean(loss_epoch), 
    #                                                                     test_error_ep, 
    #                                                                     t1-t0))