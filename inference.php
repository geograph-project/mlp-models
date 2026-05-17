<?php

require_once 'GeographModelBase.php';

function run_inference($inputFilePath) {
    if (!file_exists($inputFilePath)) {
        die("Input file not found: $inputFilePath\n");
    }

    $inputs = json_decode(file_get_contents($inputFilePath), true);
    if (!$inputs) {
        die("Invalid input JSON\n");
    }

    $allResults = [];
    $directories = array_filter(glob('*'), 'is_dir');

    foreach ($directories as $dir) {
        $modelPhp = $dir . '/model.php';
        $weightsJson = $dir . '/weights.json';

        if (file_exists($modelPhp) && file_exists($weightsJson)) {
            require_once $modelPhp;

            // Determine class name from directory name
            // e.g. gallery -> GeographGalleryModel
            // Special case for model_types -> GeographTypesModel
            $className = 'Geograph' . ucfirst(str_replace('model_', '', $dir)) . 'Model';

            if (class_exists($className)) {
                try {
                    $model = new $className($weightsJson);
                    $results = $model->predict_api($inputs);
                    $allResults = array_merge($allResults, $results);
                } catch (Exception $e) {
                    // Skip if error as requested
                    error_log("Error running model in $dir: " . $e->getMessage());
                }
            }
        }
    }

    echo json_encode($allResults, JSON_PRETTY_PRINT) . "\n";
}

if ($argc < 2) {
    echo "Usage: php inference.php <input_json_file>\n";
    exit(1);
}

run_inference($argv[1]);
