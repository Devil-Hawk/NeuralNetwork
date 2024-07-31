import java.util.Arrays;

/**
 * This class takes the instances inputs of the 7th question of the exam converts it into one hot encoding of the
 * specific input instance and outputs the one hot encoding
 * Author: Ankit Punjabi
 */
public class BreastCancerConverter {

    private static final String[] classCategories = {"no-recurrence-events", "recurrence-events"};
    private static final String[] ageCategories = {"10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"};
    private static final String[] menopauseCategories = {"lt40", "ge40", "premeno"};
    private static final String[] tumorSizeCategories = {"0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"};
    private static final String[] invNodesCategories = {"0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"};
    private static final String[] nodeCapsCategories = {"yes", "no"};
    private static final String[] degMaligCategories = {"1", "2", "3"};
    private static final String[] breastCategories = {"left", "right"};
    private static final String[] breastQuadCategories = {"left-up", "left-low", "right-up", "right-low", "central"};
    private static final String[] irradiatCategories = {"yes", "no"};

    public static void main(String[] args) {
        String[] instances = {
                "no-recurrence-events,20-29,ge40,15-19,12-14,yes,3,right,left-low,no",
                "recurrence-events,60-69,premeno,20-24,9-11,yes,3,right,left-up,yes",
                "recurrence-events,10-19,lt40,0-4,21-23,no,3,left,central,yes",
                "recurrence-events,40-49,premeno,30-34,36-39,yes,3,left,right-low,no"
        };

        try {
            convertInstances(instances);
            System.out.println("Conversion completed successfully.");
        } catch (Exception e) {
            System.err.println("Error during conversion: " + e.getMessage());
        }
    }

    private static void convertInstances(String[] instances) {
        for (int i = 0; i < instances.length; i++) {
            String[] attributes = instances[i].split(",");
            System.out.println("Instance " + (i + 1) + ": Original instance: " + Arrays.toString(attributes));

            String[] convertedAttributes = convertAttributes(attributes);
            System.out.println("Instance " + (i + 1) + ": One-hot encoded instance: " + Arrays.toString(convertedAttributes));

            System.out.println();
        }
    }

    private static String[] convertAttributes(String[] attributes) {
        String[] convertedAttributes = new String[attributes.length]; // Including the class attribute

        convertedAttributes[0] = oneHotEncodeClass(attributes[0], classCategories);
        convertedAttributes[1] = oneHotEncode(attributes[1], ageCategories);
        convertedAttributes[2] = oneHotEncode(attributes[2], menopauseCategories);
        convertedAttributes[3] = oneHotEncode(attributes[3], tumorSizeCategories);
        convertedAttributes[4] = oneHotEncode(attributes[4], invNodesCategories);
        convertedAttributes[5] = oneHotEncode(attributes[5], nodeCapsCategories);
        convertedAttributes[6] = oneHotEncode(attributes[6], degMaligCategories);
        convertedAttributes[7] = oneHotEncode(attributes[7], breastCategories);
        convertedAttributes[8] = oneHotEncode(attributes[8], breastQuadCategories);
        convertedAttributes[9] = oneHotEncode(attributes[9], irradiatCategories);

        return convertedAttributes;
    }

    private static String oneHotEncodeClass(String value, String[] categories) {
        StringBuilder encodedValue = new StringBuilder("[");
        for (String category : categories) {
            encodedValue.append(value.equals(category) ? "1" : "0").append(", ");
        }
        // Remove the trailing comma and space, and close the square bracket
        return encodedValue.substring(0, encodedValue.length() - 2) + "]";
    }

    private static String oneHotEncode(String value, String[] categories) {
        StringBuilder encodedValue = new StringBuilder("[");
        for (String category : categories) {
            encodedValue.append(value.equals(category) ? "1" : "0").append(", ");
        }
        return encodedValue.substring(0, encodedValue.length() - 2) + "]";
    }
}