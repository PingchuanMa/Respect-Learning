import tensorflow as tf
import argparse

def rename(checkpoint_dir, replace_from, replace_to, add_prefix):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            print('Renaming %s to %s.' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)

        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(description='Rename')
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--replace_from', type=str, default=None)
    parser.add_argument('--replace_to', type=str, default='')
    parser.add_argument('--add_prefix', type=str, default=None)
    args = parser.parse_args()

    rename(args.checkpoint_dir, args.replace_from, args.replace_to, args.add_prefix)


if __name__ == '__main__':
    main()