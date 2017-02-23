# RmModel
TensorFlow simple helper.

# Usage
## Define model, train and save

```
import numpy as np
def get_graph():
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('input'):
            input_data = tf.placeholder(tf.float32, shape=(None, 1), name="input_data")

        # Variable default is trainable. To save/restore, name must be provided
        with tf.name_scope('model'):
            W = tf.Variable(tf.truncated_normal([1,1], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            output_data = tf.add(input_data * W, b, name="output_data")

        # Minimize the mean squared errors.
        with tf.name_scope('train'):
            target_op = tf.placeholder(tf.float32, shape=(None, 1), name="target_op")
            loss = tf.reduce_mean(tf.square(output_data - target_op), name="loss")
            optimizer = tf.train.GradientDescentOptimizer(0.5)
            train_op = optimizer.minimize(loss, name="train_op")
        
        # plot loss curve
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()
        
    return (graph, input_data, target_op, train_op, summary_op, output_data, W, b)

graph, input_data, target_op, train_op, summary_op, output_data, W, b = get_graph()
x_data = np.random.rand(100).astype(np.float32).reshape(100,1)
y_data = x_data * 0.1 + 0.3

a = np.array([12]).astype(np.float32).reshape(-1,1)
with RmModel() as model:
    model.set_graph(graph)
    model.sample_fit(x_data, y_data, 200, train_op, summary_op, input_data, target_op)    
    model.save('./saved')
    print(model.sample_inference(a, input_data, output_data))
    print(model.session.run(W))
    print(model.session.run(b))
```

## Load model and inference
```
with RmModel() as model:
    input_op, output_op = model.load('./saved', [
        'input/input_data:0',
        'model/output_data:0'])
    print(model.sample_inference(12, input_op, output_op))
```
