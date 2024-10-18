import tensorflow as tf
@tf.keras.utils.register_keras_serializable()
class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        steering = y_pred[:, 0]  # primeira saída é para direção
        gas = y_pred[:, 1]       # segunda saída é para aceleração
        
        steering_thresholded = tf.where(steering > 1/3, 1.0, 
                                        tf.where(steering < -1/3, -1.0, 0.0))

        gas_thresholded = tf.where(gas > 1/3, 1.0, 
                                tf.where(gas < -1/3, -1.0, 0.0))
        
        #transforma num tensor denovo
        y_pred_thresholded = tf.stack([steering_thresholded, gas_thresholded], axis=1)
        
        # Calcula a quantidade de acertos
        matches = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_thresholded), dtype=tf.float32))
            
        self.correct.assign_add(matches)
        self.total.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)