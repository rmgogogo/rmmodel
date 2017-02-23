import tensorflow as tf
import os

"""
It's a simple helper class for TensorFlow.
It's mainly for quickly save/restore models.
Alternatively, you can choose Keras.
Author: RMGU
"""

class RmModel:
    """ Model helper for TensorFlow. It supports save and load models.
    """
    
    def __init__(self):
        self.model_filename = "model.pb"
        self.weights_filename = "weights"        
        self.event_folder = "/tmp/tensorboard"
        self.session = None
        self.event_writer = None        
    
    def __enter__(self):        
        return self
    
    def __exit__(self, type, value, tb):
        # For context usage (with Model() as model:)
        if self.session != None:
            self.session.close()
    
    def _check_session(self):
        if self.session == None:
            raise Exception('Wrong usage, must have a session')
    
    def set_event_folder(self, folder):
        self.event_folder = folder
    
    def save(self, folder, init_folder=True):
        """ Save graph_def and weights
        """
        
        if init_folder:
            os.system("rm -rf " + folder)
        
        self._check_session()
        self.save_graph(folder)
        self.save_weights(folder)
    
    def load(self, folder, return_elements, model_prefix=""):
        """ Import graph and weights, return the selected elements
        """
        
        # Create a new session for restore weights
        if self.session != None:
            self.session.close()
        # Here it uses InteractiveSession to bind session with current thread
        self.session = tf.InteractiveSession(graph=tf.Graph())
        graph_def = self.load_graph(folder)
        result = tf.import_graph_def(
            graph_def,
            return_elements=return_elements,
            name=model_prefix)
        # Load data
        self.load_weights(folder)        
        return result
    
    def save_graph(self, folder):
        """ Save graph definition to file
        """
        
        self._check_session()
        tf.train.write_graph(self.session.graph.as_graph_def(),
                             folder,
                             self.model_filename,
                             False)
        
    def load_graph(self, folder):
        with open(os.path.join(folder, self.model_filename), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def
        
    def save_weights(self, folder):
        """ Save parameters (weights, biases) to file
        """
        
        self._check_session()
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(self.session, os.path.join(
            folder, self.weights_filename))    

    def load_weights(self, folder):
        
        self._check_session()
        # Import meta first so that we can know the list of variables
        tf.train.import_meta_graph(os.path.join(folder,
                                                self.weights_filename + ".meta"))
        # TensorFlow API is not well designed or has bug.
        # Here we create a new saver for restore variables
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.session, os.path.join(folder, self.weights_filename))

    def open_event_writer(self):        
        if self.event_folder == None:
            pass
        
        self._check_session()
        self.event_writer = tf.summary.FileWriter(self.event_folder, self.session.graph)

    def log_summary(self, summary, step):
        if self.event_writer == None:
            pass
        
        self._check_session()
        self.event_writer.add_summary(summary, step)

    def close_event_writer(self):
        if self.event_writer == None:
            pass
        
        self._check_session()
        self.event_writer.close()

    def set_graph(self, graph):
        if self.session != None:
            self.session.close()
        # Here it uses InteractiveSession to bind session with current thread.
        self.session = tf.InteractiveSession(graph=graph)
        
    def sample_fit(self, train_x, train_y, num_step, train_op, summary_op, input_op, target_op):
        self._check_session()
        
        # TODO: batch size
        
        # Open event file
        self.open_event_writer()

        # Init
        self.session.run(tf.global_variables_initializer())

        # Train
        for step in range(num_step):
            summary, _ = self.session.run([summary_op, train_op],
                                          feed_dict={input_op: train_x, target_op: train_y})
            # Ouput events
            self.log_summary(summary, step)

        # Close event file
        self.close_event_writer()
    
    def sample_inference(self, x, input_op, output_op):
        return self.session.run(output_op, feed_dict={input_op: x})
