import { Feature } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";

export const fetchFeature = async (
  analysisName: string,
  layer: number,
  featureId: number
): Promise<Feature | null> => {
  try {
    const formattedAnalysisName = analysisName.replace("{}", layer.toString());

    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${formattedAnalysisName}/features/${featureId}`,
      {
        method: "GET",
        headers: {
          Accept: "application/x-msgpack",
        },
      }
    );

    if (!response.ok) {
      console.warn(
        `Failed to fetch feature from ${formattedAnalysisName}: ${response.status} ${response.statusText}`
      );
      return null;
    }

    const arrayBuffer = await response.arrayBuffer();
    const decoded = decode(new Uint8Array(arrayBuffer)) as Record<string, unknown>;
    const camelCased = camelcaseKeys(decoded, {
      deep: true,
      stopPaths: ["context"],
    });

    return camelCased as Feature;
  } catch (error) {
    console.error(`Error fetching feature ${featureId} from layer ${layer}:`, error);
    return null;
  }
};

/**
 * Build the dictionary name from interaction-graph metadata.
 * Uses lorsa_analysis_name / tc_analysis_name to build names like
 * BT4_lorsa_L4A_k30_e16.
 */
export const getDictionaryName = (metadata: any, layer: number, isLorsa: boolean): string => {
  if (isLorsa) {
    const analysisName = metadata?.lorsa_analysis_name;
    if (analysisName && typeof analysisName === "string") {
      if (analysisName === "BT4_lorsa") {
        return `BT4_lorsa_L${layer}A`;
      }
      if (analysisName.startsWith("BT4_lorsa_")) {
        const suffix = analysisName.replace("BT4_lorsa_", "");
        return `BT4_lorsa_L${layer}A_${suffix}`;
      }
      return analysisName.replace("{}", layer.toString());
    }
    return `BT4_lorsa_L${layer}A_k30_e16`;
  }

  const analysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
  if (analysisName && typeof analysisName === "string") {
    if (analysisName === "BT4_tc") {
      return `BT4_tc_L${layer}M`;
    }
    if (analysisName.startsWith("BT4_tc_")) {
      const suffix = analysisName.replace("BT4_tc_", "");
      return `BT4_tc_L${layer}M_${suffix}`;
    }
    return analysisName.replace("{}", layer.toString());
  }
  return `BT4_tc_L${layer}M_k30_e16`;
};
