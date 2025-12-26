import Constants from "expo-constants";
import React, {
  useMemo,
  useState,
} from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

// Configure backend base URL via Expo config:
// - Preferred: set API_BASE env var when starting Metro (see README)
// - Fallback: edit the fallback below to your Mac LAN IP if needed
const API_BASE =
  (Constants?.expoConfig?.extra?.API_BASE ||
    Constants?.manifest?.extra?.API_BASE ||
    "").trim() || "http://192.168.178.129:8000";

function MovieCard({
  item,
}) {
  return (
    <View
      style={{
        borderWidth: 1,
        borderColor:
          "#ddd",
        borderRadius: 10,
        padding: 10,
        marginBottom: 10,
      }}
    >
      <Text
        style={{
          fontSize: 14,
          fontWeight:
            "600",
        }}
      >
        {item.title}
      </Text>
      <Text
        style={{
          fontSize: 12,
          color:
            "#555",
          marginTop: 4,
        }}
      >
        {
          item.genres
        }
      </Text>
      <Text
        style={{
          fontSize: 11,
          color:
            "#999",
          marginTop: 6,
        }}
      >
        movieId:{" "}
        {
          item.movieId
        }
      </Text>
    </View>
  );
}

export default function App() {
  const [
    userId,
    setUserId,
  ] = useState(
    "155223"
  );

  const [
    demoKnownId,
    setDemoKnownId,
  ] = useState(
    "155223"
  );
  const [
    demoColdId,
    setDemoColdId,
  ] = useState(
    "999999999"
  );

  const [k, setK] =
    useState("10");
  const [
    loading,
    setLoading,
  ] =
    useState(false);
  const [
    error,
    setError,
  ] = useState("");
  const [
    data,
    setData,
  ] =
    useState(null);

  const kNum =
    useMemo(() => {
      const n =
        parseInt(
          k,
          10
        );
      return Number.isFinite(
        n
      ) &&
        n > 0 &&
        n <= 50
        ? n
        : 10;
    }, [k]);

  async function fetchDemoIds() {
    setLoading(true);
    setError("");

    try {
      const url = `${API_BASE}/demo`;
      const res = await fetch(url);
      const json = await res.json();

      if (!res.ok) {
        throw new Error(
          json?.detail ||
            `HTTP ${res.status}`
        );
      }

      const known = String(
        json?.known_user_id ??
          "155223"
      );
      const cold = String(
        json?.cold_start_user_id ??
          "999999999"
      );

      setDemoKnownId(known);
      setDemoColdId(cold);
      setUserId(known);
    } catch (e) {
      const msg = e?.message || String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  async function fetchExplain() {
    setLoading(
      true
    );
    setError("");
    setData(null);

    try {
      const uid =
        parseInt(
          userId,
          10
        );
      if (
        !Number.isFinite(
          uid
        )
      )
        throw new Error(
          "userId must be an integer"
        );

      const url = `${API_BASE}/explain/${uid}?k=${kNum}&history_n=10`;
      const res =
        await fetch(
          url
        );
      const json =
        await res.json();

      if (!res.ok) {
        throw new Error(
          json?.detail ||
            `HTTP ${res.status}`
        );
      }

      setData(json);
    } catch (e) {
      const msg = e?.message || String(e);

      // React Native fetch often throws "Network request failed" when the phone can't reach the Mac.
      if (msg.toLowerCase().includes("network request failed")) {
        setError(
          `Network request failed.\n\n` +
            `1) Ensure FastAPI is running with --host 0.0.0.0\n` +
            `2) Ensure phone + Mac are on the same Wi‑Fi\n` +
            `3) Set API_BASE to your Mac LAN IP (not 127.0.0.1)\n\n` +
            `Current API_BASE: ${API_BASE}`
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(
        false
      );
    }
  }

  const historyItems = data?.history?.items || [];

  // Prefer /explain shape: data.als.items / data.diverse.items
  // Fallback to /recommend/compare shape: als_items / diverse_items
  const alsItems =
    data?.als?.items ||
    data?.als_items ||
    [];
  const diverseItems =
    data?.diverse?.items ||
    data?.diverse_items ||
    [];

  return (
    <SafeAreaView
      style={{
        flex: 1,
        backgroundColor:
          "#fff",
      }}
    >
      <View
        style={{
          padding: 16,
          borderBottomWidth: 1,
          borderBottomColor:
            "#eee",
        }}
      >
        <Text
          style={{
            fontSize: 18,
            fontWeight:
              "700",
          }}
        >
          Movie Recs
          Demo
        </Text>
        <Text
          style={{
            fontSize: 12,
            color:
              "#666",
            marginTop: 4,
          }}
        >
          Explain: history + ALS vs Diverse
        </Text>

        <View
          style={{
            flexDirection:
              "row",
            gap: 10,
            marginTop: 12,
          }}
        >
          <View
            style={{
              flex: 1,
            }}
          >
            <Text
              style={{
                fontSize: 12,
                color:
                  "#444",
                marginBottom: 6,
              }}
            >
              userId
            </Text>
            <TextInput
              value={
                userId
              }
              onChangeText={
                setUserId
              }
              keyboardType="number-pad"
              style={{
                borderWidth: 1,
                borderColor:
                  "#ddd",
                borderRadius: 10,
                padding: 10,
              }}
            />
          </View>

          <View
            style={{
              width: 90,
            }}
          >
            <Text
              style={{
                fontSize: 12,
                color:
                  "#444",
                marginBottom: 6,
              }}
            >
              k
            </Text>
            <TextInput
              value={
                k
              }
              onChangeText={
                setK
              }
              keyboardType="number-pad"
              style={{
                borderWidth: 1,
                borderColor:
                  "#ddd",
                borderRadius: 10,
                padding: 10,
              }}
            />
          </View>
        </View>

        <View style={{ flexDirection: "row", gap: 10, marginTop: 10 }}>
          <Pressable
            onPress={fetchDemoIds}
            style={({ pressed }) => ({
              flex: 1,
              backgroundColor: pressed ? "#222" : "#111",
              paddingVertical: 10,
              borderRadius: 12,
              alignItems: "center",
            })}
          >
            <Text style={{ fontWeight: "700", color: "#fff" }}>
              Load demo IDs
            </Text>
          </Pressable>

          <Pressable
            onPress={() => setUserId(demoKnownId)}
            style={({ pressed }) => ({
              flex: 1,
              backgroundColor: pressed ? "#e6e6e6" : "#f2f2f2",
              paddingVertical: 10,
              borderRadius: 12,
              alignItems: "center",
            })}
          >
            <Text style={{ fontWeight: "700" }}>Use known</Text>
          </Pressable>

          <Pressable
            onPress={() => setUserId(demoColdId)}
            style={({ pressed }) => ({
              flex: 1,
              backgroundColor: pressed ? "#e6e6e6" : "#f2f2f2",
              paddingVertical: 10,
              borderRadius: 12,
              alignItems: "center",
            })}
          >
            <Text style={{ fontWeight: "700" }}>Use cold-start</Text>
          </Pressable>
        </View>

        <Pressable
          onPress={
            fetchExplain
          }
          style={{
            marginTop: 12,
            backgroundColor:
              "#111",
            paddingVertical: 12,
            borderRadius: 12,
            alignItems:
              "center",
          }}
        >
          <Text
            style={{
              color:
                "#fff",
              fontWeight:
                "700",
            }}
          >
            {loading
              ? "Loading..."
              : "Explain + recommend"}
          </Text>
        </Pressable>

        {error ? (
          <Text
            style={{
              marginTop: 10,
              color:
                "#b00020",
            }}
          >
            {error}
          </Text>
        ) : null}
      </View>

      {loading ? (
        <View
          style={{
            flex: 1,
            alignItems:
              "center",
            justifyContent:
              "center",
          }}
        >
          <ActivityIndicator />
        </View>
      ) : (
        <ScrollView
          style={{
            flex: 1,
          }}
          contentContainerStyle={{
            padding: 16,
          }}
        >
          <View style={{ marginBottom: 16 }}>
            <Text style={{ fontSize: 14, fontWeight: "800", marginBottom: 10 }}>
              Because you liked
            </Text>

            {historyItems.length === 0 ? (
              <Text style={{ fontSize: 12, color: "#666" }}>
                No training history found (cold start) — using popularity fallback.
              </Text>
            ) : (
              historyItems.map((it) => (
                <MovieCard key={`hist-${it.movieId}`} item={it} />
              ))
            )}
          </View>
          <View
            style={{
              flexDirection:
                "row",
              gap: 12,
            }}
          >
            <View
              style={{
                flex: 1,
              }}
            >
              <Text
                style={{
                  fontSize: 14,
                  fontWeight:
                    "800",
                  marginBottom: 4,
                }}
              >
                ALS
              </Text>
              <Text style={{ fontSize: 12, color: "#666", marginBottom: 10 }}>
                Accuracy / safe picks
              </Text>
              {alsItems.map(
                (
                  it
                ) => (
                  <MovieCard
                    key={`als-${it.movieId}`}
                    item={
                      it
                    }
                  />
                )
              )}
            </View>

            <View
              style={{
                flex: 1,
              }}
            >
              <Text
                style={{
                  fontSize: 14,
                  fontWeight:
                    "800",
                  marginBottom: 4,
                }}
              >
                Diverse
              </Text>
              <Text style={{ fontSize: 12, color: "#666", marginBottom: 10 }}>
                Discovery / less popular
              </Text>
              {diverseItems.map(
                (
                  it
                ) => (
                  <MovieCard
                    key={`div-${it.movieId}`}
                    item={
                      it
                    }
                  />
                )
              )}
            </View>
          </View>
        </ScrollView>
      )}
    </SafeAreaView>
  );
}
