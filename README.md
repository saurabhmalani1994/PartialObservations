# PartialObservations

train_model.py is the script to run to train a neural network

utils.py contains ALL the functions

    class CSTRDataset creates the training dataset. The preprocessing needed to work with incommensurate times happens HERE, self.time contains the 'key' time points (where loss will be evaluated), AND 'intermediate' time points (for the numerical integrator).
            If data is present, the value will be a positive value (both variables physically are strictly positive). If data is NOT present at that time point, the data point at that value will be -1. For 'key' time points, at least ONE of the variables will be >0 at that time point. For 'intermediate' time points, BOTH variables will be -1.
    
    class Network is the neural network. The 'Settings' sub part is where you can choose the numerical integrator template to use (as well as grey/black box but ignore that for now)
            refer more to the forward_manual method. The torch.where is how the model chooses what data to feed to the network depending if (1) we're training autoregressively or with teacher forcing, and/or (2) if the real data is present or not.
    
    class myModel is what does the actual training. This is super messy right now (had to do a lot of trial and error to get good convergence sometimes)
            To do training, on the same dataset I obtain loss twice - once with 'full teacher forcing' and once with 'autoregressive integration'. self.warmup dictates HOW autoregressive the latter is. I then take a log sum of the two losses, and self.alpha controls the relative weight of the two losses. In my experience, keeping self.warmup at 0.0 and self.alpha at 0.5 works relatively well.
            In training, I first train for the first 200 epochs with a constant high learning rate with FULL teacher forcing ONLY to bring the network to a stable manifold where full autoregressive training won't blow up. For the remainder of training, I then do the balanced teacher-forcing/autoregressive training with the OneCycleLR learning rate/momentum hyperparameter scheduler.
            The validation loss is always computed with ONLY autoregressive predictions (hence it'll often be significantly higher than the training loss) because I use this to judge if the network is actually learning long term behavior correctly.
            LOSS is only computed for the time points where real data is present, if real data is NOT present, it compares 0 against 0 i.e. no loss. I use a custom loss function my_Loss that when taking the mean loss over all data points only divides by the number of data points that have real data, so that the number of '0 vs 0' data points does not artificially bring down the mean loss. Doing so keeps both variables equivalently weighted: if x1 has 10 data points and x2 and 100, the AVERAGE of the loss at 10 x1 data points is added to the AVERAGE of the loss at 100 x2 data points for the total loss. Additionally, both variables are first NORMALIZED between 0 and 1 before loss is computed to further ensure both variables are equivalently weighted (otherwise x2 is about an order of magnitude larger than x1). self.x1_loss_mult and self.x2_loss_mult can allow us to favor one variable over the other if needed, currently both are set to 1.
            
            
=========================================================================================================
            
            
To Train Model and Create Plots:

run train_model.py
run make_plots.py

Figures will be created in Figures folder
