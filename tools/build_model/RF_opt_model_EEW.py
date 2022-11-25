import tensorflow as tf
import numpy as np


@tf.function
def distributed_train_step_RF_fusion(strategy, train_args):
    def train_step_gpus(model, opt, global_batch_size,
        train_trc, train_spec, train_ps, train_EQmask, 
        loss_estimator, class_weight):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
            # make predictions and estimate loss
            train_pred_ps, train_pred_EQmask = model(
                [train_trc, train_spec], training=True)
            per_replica_losses_ps = \
                loss_estimator(train_ps, train_pred_ps)
            per_replica_losses_EQmask = \
                loss_estimator(train_EQmask, train_pred_EQmask)

            per_replica_losses = \
                per_replica_losses_ps * class_weight[0] +\
                per_replica_losses_EQmask * class_weight[1]
            grad_loss = tf.reduce_sum(per_replica_losses)/global_batch_size
        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return per_replica_losses, per_replica_losses_ps, \
            per_replica_losses_EQmask

    per_replica_losses, per_replica_losses_ps, \
        per_replica_losses_EQmask = \
        strategy.run(
            train_step_gpus, args=train_args)

    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None )                 
    mean_batch_loss_ps = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_ps, axis=None )
    mean_batch_loss_EQmask = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses_EQmask, axis=None ) 
    #train_loss_avg.update_state(mean_loss)
    return mean_batch_loss, mean_batch_loss_ps, \
        mean_batch_loss_EQmask

@tf.function
def distributed_val_step_RF_fusion(strategy, val_args):
    def val_step_gpus(
        model, global_batch_size,
        val_trc, val_spec, val_ps, val_EQmask,
        loss_estimator, class_weight):
        # estimate validation data 
        val_pred_ps, val_pred_EQmask =\
            model([val_trc, val_spec], training=False)
        per_replica_losses_ps = loss_estimator(val_ps, val_pred_ps)
        per_replica_losses_EQmask = loss_estimator(val_EQmask, val_pred_EQmask)

        per_replica_losses = \
            per_replica_losses_ps*class_weight[0] +\
            per_replica_losses_EQmask*class_weight[1]

        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
        per_replica_losses, axis=None)
    return mean_batch_loss

if __name__=='__main__':
    pass