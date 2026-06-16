import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View
} from "react-native";

const DEFAULT_API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL || "http://16.16.171.242:8000";

function prettyClassName(className) {
  if (!className) {
    return "-";
  }
  return className.replaceAll("_", " ");
}

function percent(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

function fileNameFromUri(uri) {
  const fallback = "plant-photo.jpg";
  if (!uri) {
    return fallback;
  }
  return uri.split("/").pop() || fallback;
}

function mimeTypeFromUri(uri) {
  const name = fileNameFromUri(uri).toLowerCase();
  if (name.endsWith(".png")) {
    return "image/png";
  }
  if (name.endsWith(".webp")) {
    return "image/webp";
  }
  return "image/jpeg";
}

export default function App() {
  const [apiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [removeBackground, setRemoveBackground] = useState(true);
  const [multiCrop, setMultiCrop] = useState(true);
  const [maxCrops, setMaxCrops] = useState(8);

  const diagnosisUrl = useMemo(() => {
    const trimmed = apiBaseUrl.trim().replace(/\/$/, "");
    const cropCount = multiCrop ? maxCrops : 1;
    return `${trimmed}/diagnose?remove_background=${removeBackground}&multi_crop=${multiCrop}&max_crops=${cropCount}&top_k=3`;
  }, [apiBaseUrl, maxCrops, multiCrop, removeBackground]);

  function changeMaxCrops(delta) {
    setMaxCrops((current) => Math.min(16, Math.max(1, current + delta)));
  }

  async function pickImage() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission required", "Allow gallery access to select an image.");
      return;
    }

    const selected = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
      allowsEditing: false
    });

    if (!selected.canceled && selected.assets?.length) {
      setImage(selected.assets[0]);
      setResult(null);
    }
  }

  async function takePhoto() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission required", "Allow camera access to photograph the plant.");
      return;
    }

    const captured = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
      allowsEditing: false
    });

    if (!captured.canceled && captured.assets?.length) {
      setImage(captured.assets[0]);
      setResult(null);
    }
  }

  async function diagnoseImage() {
    if (!image?.uri) {
      Alert.alert("Select an image", "Select or photograph a leaf before analysis.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", {
        uri: image.uri,
        name: fileNameFromUri(image.uri),
        type: image.mimeType || mimeTypeFromUri(image.uri)
      });

      const response = await fetch(diagnosisUrl, {
        method: "POST",
        headers: {
          Accept: "application/json"
        },
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "The server could not process the image.");
      }

      setResult(payload);
      setHistory((current) => [
        {
          id: `${Date.now()}`,
          className: payload.prediction?.class_name,
          confidence: payload.prediction?.confidence,
          createdAt: new Date().toLocaleString()
        },
        ...current.slice(0, 4)
      ]);
    } catch (error) {
      Alert.alert("Diagnosis error", error.message);
    } finally {
      setLoading(false);
    }
  }

  const prediction = result?.prediction;

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.page}>
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>Plant Diagnosis</Text>
          </View>
        </View>

        <View style={styles.panel}>
          <View style={styles.panelTitleRow}>
            <Ionicons name="options-outline" size={20} color="#1b6b5c" />
            <Text style={styles.sectionTitleCompact}>Analysis settings</Text>
          </View>

          <View style={styles.settingRow}>
            <View style={styles.settingLabelWrap}>
              <Text style={styles.settingLabel}>Remove background</Text>
              <Text style={styles.settingValue}>{removeBackground ? "On" : "Off"}</Text>
            </View>
            <Switch
              value={removeBackground}
              onValueChange={setRemoveBackground}
              trackColor={{ false: "#cfdad5", true: "#9fd7c5" }}
              thumbColor={removeBackground ? "#1b6b5c" : "#f7faf8"}
            />
          </View>

          <View style={styles.settingRow}>
            <View style={styles.settingLabelWrap}>
              <Text style={styles.settingLabel}>Multi-crop voting</Text>
              <Text style={styles.settingValue}>{multiCrop ? "On" : "Full image"}</Text>
            </View>
            <Switch
              value={multiCrop}
              onValueChange={setMultiCrop}
              trackColor={{ false: "#cfdad5", true: "#9fd7c5" }}
              thumbColor={multiCrop ? "#1b6b5c" : "#f7faf8"}
            />
          </View>

          <View style={[styles.settingRow, !multiCrop && styles.settingDisabled]}>
            <View style={styles.settingLabelWrap}>
              <Text style={styles.settingLabel}>Crops</Text>
              <Text style={styles.settingValue}>{multiCrop ? "1 - 16" : "Disabled"}</Text>
            </View>
            <View style={styles.stepper}>
              <TouchableOpacity
                style={[styles.stepButton, (!multiCrop || maxCrops <= 1) && styles.stepButtonDisabled]}
                onPress={() => changeMaxCrops(-1)}
                disabled={!multiCrop || maxCrops <= 1}
              >
                <Ionicons name="remove" size={18} color={multiCrop && maxCrops > 1 ? "#17443b" : "#8aa097"} />
              </TouchableOpacity>
              <Text style={styles.cropCount}>{multiCrop ? maxCrops : 1}</Text>
              <TouchableOpacity
                style={[styles.stepButton, (!multiCrop || maxCrops >= 16) && styles.stepButtonDisabled]}
                onPress={() => changeMaxCrops(1)}
                disabled={!multiCrop || maxCrops >= 16}
              >
                <Ionicons name="add" size={18} color={multiCrop && maxCrops < 16 ? "#17443b" : "#8aa097"} />
              </TouchableOpacity>
            </View>
          </View>
        </View>

        <View style={styles.actions}>
          <TouchableOpacity style={styles.actionButton} onPress={takePhoto}>
            <Ionicons name="camera-outline" size={22} color="#ffffff" />
            <Text style={styles.actionButtonText}>Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.secondaryButton} onPress={pickImage}>
            <Ionicons name="image-outline" size={22} color="#17443b" />
            <Text style={styles.secondaryButtonText}>Gallery</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.previewBox}>
          {image?.uri ? (
            <Image source={{ uri: image.uri }} style={styles.previewImage} />
          ) : (
            <View style={styles.emptyPreview}>
              <Ionicons name="leaf-outline" size={42} color="#6b8f83" />
              <Text style={styles.emptyPreviewText}>Select or photograph a leaf</Text>
            </View>
          )}
        </View>

        <TouchableOpacity
          style={[styles.diagnoseButton, (!image || loading) && styles.disabledButton]}
          onPress={diagnoseImage}
          disabled={!image || loading}
        >
          {loading ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <>
              <Ionicons name="analytics-outline" size={22} color="#ffffff" />
              <Text style={styles.diagnoseButtonText}>Analyze image</Text>
            </>
          )}
        </TouchableOpacity>

        {prediction && (
          <View style={styles.resultPanel}>
            <Text style={styles.sectionTitle}>Diagnosis</Text>
            <Text style={styles.predictionName}>{prettyClassName(prediction.class_name)}</Text>
            <Text style={styles.confidence}>Confidence: {percent(prediction.confidence)}</Text>
            <Text style={styles.explanation}>{prediction.explanation_text}</Text>

            <View style={styles.pipelineBox}>
              <View style={styles.pipelineItem}>
                <Ionicons
                  name={result.pipeline?.preprocessing?.includes("background") ? "cut-outline" : "image-outline"}
                  size={18}
                  color="#1b6b5c"
                />
                <Text style={styles.pipelineText}>
                  Background: {result.pipeline?.preprocessing?.includes("background") ? "removed" : "original"}
                </Text>
              </View>
              <View style={styles.pipelineItem}>
                <Ionicons
                  name={result.pipeline?.crop_generation === "multi_crop" ? "grid-outline" : "scan-outline"}
                  size={18}
                  color="#1b6b5c"
                />
                <Text style={styles.pipelineText}>
                  Analysis: {result.pipeline?.crop_generation === "multi_crop" ? `${result.crops?.length || 0} crops` : "full image"}
                </Text>
              </View>
            </View>

            {result.alternatives?.length > 0 && (
              <View style={styles.subsection}>
                <Text style={styles.subsectionTitle}>Alternative</Text>
                {result.alternatives.map((item) => (
                  <View key={item.class_name} style={styles.row}>
                    <Text style={styles.rowText}>{prettyClassName(item.class_name)}</Text>
                    <Text style={styles.rowValue}>{percent(item.confidence)}</Text>
                  </View>
                ))}
              </View>
            )}

            {result.crops?.length > 0 && (
              <View style={styles.subsection}>
                <Text style={styles.subsectionTitle}>Crop votes</Text>
                {result.crops.slice(0, 5).map((crop) => (
                  <View key={`${crop.crop_index}-${crop.source}`} style={styles.row}>
                    <Text style={styles.rowText}>
                      #{crop.crop_index + 1} {prettyClassName(crop.predicted_class)}
                    </Text>
                    <Text style={styles.rowValue}>{percent(crop.confidence)}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {history.length > 0 && (
          <View style={styles.panel}>
            <Text style={styles.sectionTitle}>Local history</Text>
            {history.map((item) => (
              <View key={item.id} style={styles.historyItem}>
                <View>
                  <Text style={styles.historyClass}>{prettyClassName(item.className)}</Text>
                  <Text style={styles.historyDate}>{item.createdAt}</Text>
                </View>
                <Text style={styles.historyConfidence}>{percent(item.confidence)}</Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#f5f7f4"
  },
  page: {
    padding: 18,
    paddingBottom: 32
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 18
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: "#12312b"
  },
  panel: {
    padding: 14,
    borderRadius: 8,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#dbe5df",
    marginBottom: 14
  },
  panelTitleRow: {
    minHeight: 28,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8
  },
  sectionTitleCompact: {
    fontSize: 16,
    fontWeight: "800",
    color: "#12312b"
  },
  settingRow: {
    minHeight: 56,
    borderTopWidth: 1,
    borderTopColor: "#e6eee9",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12
  },
  settingDisabled: {
    opacity: 0.55
  },
  settingLabelWrap: {
    flex: 1,
    minWidth: 0
  },
  settingLabel: {
    color: "#263b36",
    fontSize: 15,
    fontWeight: "800"
  },
  settingValue: {
    marginTop: 2,
    color: "#6c827a",
    fontSize: 12,
    fontWeight: "700"
  },
  stepper: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#cad8d2",
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "#f6faf8"
  },
  stepButton: {
    width: 38,
    height: 38,
    alignItems: "center",
    justifyContent: "center"
  },
  stepButtonDisabled: {
    backgroundColor: "#edf3f0"
  },
  cropCount: {
    width: 38,
    textAlign: "center",
    color: "#12312b",
    fontSize: 16,
    fontWeight: "900"
  },
  actions: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 14
  },
  actionButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: "#1b6b5c",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8
  },
  actionButtonText: {
    color: "#ffffff",
    fontWeight: "800",
    fontSize: 16
  },
  secondaryButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: "#e5eee9",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8
  },
  secondaryButtonText: {
    color: "#17443b",
    fontWeight: "800",
    fontSize: 16
  },
  previewBox: {
    minHeight: 260,
    borderRadius: 8,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#d5e0da",
    backgroundColor: "#ffffff",
    marginBottom: 14
  },
  previewImage: {
    width: "100%",
    height: 300,
    resizeMode: "cover"
  },
  emptyPreview: {
    minHeight: 260,
    alignItems: "center",
    justifyContent: "center",
    gap: 10
  },
  emptyPreviewText: {
    color: "#6b8f83",
    fontWeight: "700"
  },
  diagnoseButton: {
    minHeight: 52,
    borderRadius: 8,
    backgroundColor: "#243f39",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
    marginBottom: 16
  },
  disabledButton: {
    opacity: 0.55
  },
  diagnoseButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "800"
  },
  resultPanel: {
    padding: 16,
    borderRadius: 8,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#dbe5df",
    marginBottom: 14
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#12312b",
    marginBottom: 10
  },
  predictionName: {
    fontSize: 24,
    fontWeight: "900",
    color: "#1b6b5c",
    textTransform: "capitalize"
  },
  confidence: {
    marginTop: 4,
    color: "#526b63",
    fontWeight: "700"
  },
  explanation: {
    marginTop: 12,
    color: "#263b36",
    lineHeight: 21
  },
  pipelineBox: {
    marginTop: 14,
    padding: 12,
    borderRadius: 8,
    backgroundColor: "#eef7f3",
    borderWidth: 1,
    borderColor: "#d2e7de",
    gap: 8
  },
  pipelineItem: {
    minHeight: 28,
    flexDirection: "row",
    alignItems: "center",
    gap: 8
  },
  pipelineText: {
    flex: 1,
    color: "#17443b",
    fontWeight: "800"
  },
  subsection: {
    marginTop: 16
  },
  subsectionTitle: {
    color: "#12312b",
    fontWeight: "800",
    marginBottom: 8
  },
  row: {
    minHeight: 38,
    borderTopWidth: 1,
    borderTopColor: "#e6eee9",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  rowText: {
    flex: 1,
    color: "#304b44",
    textTransform: "capitalize"
  },
  rowValue: {
    color: "#1b6b5c",
    fontWeight: "800"
  },
  historyItem: {
    minHeight: 48,
    borderTopWidth: 1,
    borderTopColor: "#e6eee9",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  historyClass: {
    color: "#263b36",
    fontWeight: "800",
    textTransform: "capitalize"
  },
  historyDate: {
    color: "#6c827a",
    fontSize: 12
  },
  historyConfidence: {
    color: "#1b6b5c",
    fontWeight: "900"
  }
});
