import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class BertInference {

  public static void main(String[] args) {
    Session session = SavedModelBundle.load(args[0], "serve").session();

    System.out.println("sss");
    long[] input_shape = new long[]{2, 64};
    long[] label_shape = new long[]{2};

    IntBuffer query_num_buf = IntBuffer.allocate(2);
    IntBuffer random_int = IntBuffer.allocate(128);

    for (int i=0; i < 128; i++) {
        random_int.put(1);
    }
    query_num_buf.put(0);
    query_num_buf.put(1);

    random_int.flip();
    query_num_buf.flip();

    Tensor<Integer> input_ids = Tensor.create(input_shape, random_int);
    Tensor<Integer> input_mask = Tensor.create(input_shape, random_int);
    Tensor<Integer> segment_ids = Tensor.create(input_shape, random_int);
    Tensor<Integer> label_ids = Tensor.create(label_shape, query_num_buf);

    // Doesn't look like Java has a good way to convert the
    // input/output name ("x", "scores") to their underlying tensor,
    // so we hard code them ("Placeholder:0", ...).
    // You can inspect them on the command-line with saved_model_cli:
    //
    // $ saved_model_cli show --dir $EXPORT_DIR --tag_set serve --signature_def serving_default
    final String input_ids_name = "input_ids:0";
    final String input_mask_name = "input_mask:0";
    final String segment_ids_name = "segment_ids:0";
    final String label_ids_name = "label_ids:0";
    final String scoresName = "Softmax:0";

    Tensor<Float> outputs = session.runner()
        .feed(input_ids_name, input_ids)
        .feed(input_mask_name, input_mask)
        .feed(segment_ids_name, segment_ids)
        .feed(label_ids_name, label_ids)
        .fetch(scoresName)
        .run()
        .get(0).expect(Float.class);

    // Outer dimension is batch size; inner dimension is number of classes
    float[][] scores = new float[2][3];
    outputs.copyTo(scores);
    System.out.println(Arrays.deepToString(scores));
  }
}
