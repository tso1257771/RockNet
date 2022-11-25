import tensorflow as tf
import numpy as np
'''
Sources:
https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/

# Encapsulating the forward and backward passs of data using
# tf.GradientTape for updating model weights.

'''
#--  distributed training

@tf.function
def distributed_train_step(strategy, train_args):
    def train_step_gpus(model, opt, global_batch_size,
        train_trc, train_ps, train_EQmask, train_EQocc,
        loss_estimator, class_weight):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
            # make predictions and estimate loss
            train_pred_ps, train_pred_EQmask, train_pred_EQocc = \
                model(train_trc, training=True)

            per_replica_losses_ps = \
                loss_estimator(train_ps, train_pred_ps)
            per_replica_losses_EQmask = \
                loss_estimator(train_EQmask, train_pred_EQmask)
            per_replica_losses_EQocc = \
                loss_estimator(train_EQocc, train_pred_EQocc)

            per_replica_losses = \
                tf.reduce_sum(
                    per_replica_losses_ps * class_weight[0], axis=1) +\
                tf.reduce_sum(
                    per_replica_losses_EQmask * class_weight[1], axis=1)+\
                per_replica_losses_EQocc * class_weight[2]

            grad_loss = tf.reduce_sum(per_replica_losses)/global_batch_size
        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return per_replica_losses, per_replica_losses_ps, \
            per_replica_losses_EQmask, per_replica_losses_EQocc

    per_replica_losses, per_replica_losses_ps, \
        per_replica_losses_EQmask, per_replica_losses_EQocc = \
        strategy.run(
            train_step_gpus, args=train_args)

    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None )                 
    mean_batch_loss_ps = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_ps, axis=None )
    mean_batch_loss_EQmask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQmask, axis=None ) 
    mean_batch_loss_EQocc = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQocc, axis=None )         
    #train_loss_avg.update_state(mean_loss)
    return mean_batch_loss, mean_batch_loss_ps, \
        mean_batch_loss_EQmask, mean_batch_loss_EQocc

@tf.function
def distributed_val_step(strategy, val_args):
    def val_step_gpus(
        model, global_batch_size,
        val_trc, val_ps, val_EQmask, val_EQocc,
        loss_estimator, class_weight):
        # estimate validation data 
        val_pred_ps, val_pred_EQmask, val_pred_EQocc =\
            model(val_trc, training=False)
        per_replica_losses_ps = loss_estimator(val_ps, val_pred_ps)
        per_replica_losses_EQmask = loss_estimator(val_EQmask, val_pred_EQmask)
        per_replica_losses_EQocc = loss_estimator(val_EQocc, val_pred_EQocc)
        per_replica_losses = \
            tf.reduce_sum(per_replica_losses_ps*class_weight[0],
                axis=1) +\
            tf.reduce_sum(per_replica_losses_EQmask*class_weight[1],
                axis=1) +\
            per_replica_losses_EQocc*class_weight[2]

        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None)
    return mean_batch_loss


@tf.function
def distributed_train_step_RF_fusion(strategy, train_args):
    def train_step_gpus(model, opt, global_batch_size,
        train_trc, train_spec, train_ps, train_EQmask, train_RFmask,
        train_EQocc, train_RFocc,
        loss_estimator, class_weight):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
            # make predictions and estimate loss
            train_pred_ps, train_pred_EQmask, train_pred_RFmask,\
                train_pred_EQocc, train_pred_RFocc = \
                model([train_trc, train_spec], training=True)

            per_replica_losses_ps = \
                loss_estimator(train_ps, train_pred_ps)
            per_replica_losses_EQmask = \
                loss_estimator(train_EQmask, train_pred_EQmask)
            per_replica_losses_EQocc = \
                loss_estimator(train_EQocc, train_pred_EQocc)
            per_replica_losses_RFmask = \
                loss_estimator(train_RFmask, train_pred_RFmask)
            per_replica_losses_RFocc = \
                loss_estimator(train_RFocc, train_pred_RFocc)

            per_replica_losses = \
                tf.reduce_sum(
                    per_replica_losses_ps * class_weight[0], axis=1) +\
                tf.reduce_sum(
                    per_replica_losses_EQmask * class_weight[1], axis=1)+\
                tf.reduce_sum(
                    per_replica_losses_RFmask * class_weight[2], axis=1)+\
                per_replica_losses_EQocc * class_weight[3]+\
                per_replica_losses_RFocc * class_weight[4]

            grad_loss = tf.reduce_sum(per_replica_losses)/global_batch_size
        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return per_replica_losses, per_replica_losses_ps, \
            per_replica_losses_EQmask, per_replica_losses_RFmask,\
            per_replica_losses_EQocc, per_replica_losses_RFocc

    per_replica_losses, per_replica_losses_ps, \
        per_replica_losses_EQmask, per_replica_losses_RFmask,\
        per_replica_losses_EQocc, per_replica_losses_RFocc = \
        strategy.run(
            train_step_gpus, args=train_args)

    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None )                 
    mean_batch_loss_ps = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_ps, axis=None )
    mean_batch_loss_EQmask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQmask, axis=None ) 
    mean_batch_loss_EQocc = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQocc, axis=None )
    mean_batch_loss_RFmask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_RFmask, axis=None ) 
    mean_batch_loss_RFocc = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_RFocc, axis=None )         

    return mean_batch_loss, mean_batch_loss_ps, \
        mean_batch_loss_EQmask, mean_batch_loss_RFmask,\
        mean_batch_loss_EQocc, mean_batch_loss_RFocc

@tf.function
def distributed_val_step_RF_fusion(strategy, val_args):
    def val_step_gpus(
        model, global_batch_size,
        val_trc, val_spec, val_ps, val_EQmask, val_RFmask,
        val_EQocc, val_RFocc,
        loss_estimator, class_weight):
        # estimate validation data 
        val_pred_ps, val_pred_EQmask, val_pred_RFmask, \
            val_pred_EQocc, val_pred_RFocc =\
            model([val_trc, val_spec], training=False)
        per_replica_losses_ps = loss_estimator(val_ps, val_pred_ps)
        per_replica_losses_EQmask = loss_estimator(val_EQmask, val_pred_EQmask)
        per_replica_losses_EQocc = loss_estimator(val_EQocc, val_pred_EQocc)
        per_replica_losses_RFmask = loss_estimator(val_RFmask, val_pred_RFmask)
        per_replica_losses_RFocc = loss_estimator(val_RFocc, val_pred_RFocc)        
        per_replica_losses = \
            tf.reduce_sum(per_replica_losses_ps*class_weight[0],
                axis=1) +\
            tf.reduce_sum(per_replica_losses_EQmask*class_weight[1],
                axis=1) +\
            tf.reduce_sum(per_replica_losses_RFmask*class_weight[2],
                axis=1) +\
            per_replica_losses_EQocc*class_weight[3] +\
            per_replica_losses_RFocc*class_weight[4]

        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None)
    return mean_batch_loss

@tf.function
def distributed_train_step_RF(strategy, train_args):
    def train_step_gpus(model, opt, global_batch_size,
        train_trc, train_ps, train_EQmask, train_RFmask,
        train_EQocc, train_RFocc,
        loss_estimator, class_weight):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
            # make predictions and estimate loss
            train_pred_ps, train_pred_EQmask, train_pred_RFmask,\
                train_pred_EQocc, train_pred_RFocc = \
                    model(train_trc, training=True)

            per_replica_losses_ps = \
                loss_estimator(train_ps, train_pred_ps)
            per_replica_losses_EQmask = \
                loss_estimator(train_EQmask, train_pred_EQmask)
            per_replica_losses_EQocc = \
                loss_estimator(train_EQocc, train_pred_EQocc)
            per_replica_losses_RFmask = \
                loss_estimator(train_RFmask, train_pred_RFmask)
            per_replica_losses_RFocc = \
                loss_estimator(train_RFocc, train_pred_RFocc)

            per_replica_losses = \
                tf.reduce_sum(
                    per_replica_losses_ps * class_weight[0], axis=1) +\
                tf.reduce_sum(
                    per_replica_losses_EQmask * class_weight[1], axis=1)+\
                tf.reduce_sum(
                    per_replica_losses_RFmask * class_weight[2], axis=1)+\
                per_replica_losses_EQocc * class_weight[3]+\
                per_replica_losses_RFocc * class_weight[4]

            grad_loss = tf.reduce_sum(per_replica_losses)/global_batch_size
        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return per_replica_losses, per_replica_losses_ps, \
            per_replica_losses_EQmask, per_replica_losses_RFmask,\
            per_replica_losses_EQocc, per_replica_losses_RFocc

    per_replica_losses, per_replica_losses_ps, \
        per_replica_losses_EQmask, per_replica_losses_RFmask,\
        per_replica_losses_EQocc, per_replica_losses_RFocc = \
        strategy.run(
            train_step_gpus, args=train_args)

    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None )                 
    mean_batch_loss_ps = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_ps, axis=None )
    mean_batch_loss_EQmask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQmask, axis=None ) 
    mean_batch_loss_EQocc = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQocc, axis=None )
    mean_batch_loss_RFmask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_RFmask, axis=None ) 
    mean_batch_loss_RFocc = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_RFocc, axis=None )              

    return mean_batch_loss, mean_batch_loss_ps, \
        mean_batch_loss_EQmask, mean_batch_loss_RFmask,\
        mean_batch_loss_EQocc, mean_batch_loss_RFocc

@tf.function
def distributed_val_step_RF(strategy, val_args):
    def val_step_gpus(
        model, global_batch_size,
        val_trc, val_ps, val_EQmask, val_RFmask, val_EQocc, val_RFocc,
        loss_estimator, class_weight):
        # estimate validation data 
        val_pred_ps, val_pred_EQmask, val_pred_RFmask, \
            val_pred_EQocc, val_pred_RFocc =\
            model(val_trc, training=False)
        per_replica_losses_ps = loss_estimator(val_ps, val_pred_ps)
        per_replica_losses_EQmask = loss_estimator(val_EQmask, val_pred_EQmask)
        per_replica_losses_EQocc = loss_estimator(val_EQocc, val_pred_EQocc)
        per_replica_losses_RFmask = loss_estimator(val_RFmask, val_pred_RFmask)
        per_replica_losses_RFocc = loss_estimator(val_RFocc, val_pred_RFocc)        
        per_replica_losses = \
            tf.reduce_sum(per_replica_losses_ps*class_weight[0],
                axis=1) +\
            tf.reduce_sum(per_replica_losses_EQmask*class_weight[1],
                axis=1) +\
            tf.reduce_sum(per_replica_losses_RFmask*class_weight[2],
                axis=1) +\
            per_replica_losses_EQocc*class_weight[3] +\
            per_replica_losses_RFocc*class_weight[4]

        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None)
    return mean_batch_loss

if __name__=='__main__':
    pass