import React, { useEffect } from 'react';
import { View, Text, StatusBar, Animated, Image } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

export default function SplashScreen({ onFinish }) {
  const fadeAnim = new Animated.Value(0);
  const scaleAnim = new Animated.Value(0.8);

  useEffect(() => {
    // Fade in and scale animation
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 4,
        useNativeDriver: true,
      }),
    ]).start();

    // Auto-hide splash screen after 2.5 seconds
    const timer = setTimeout(() => {
      if (onFinish) onFinish();
    }, 2500);

    return () => clearTimeout(timer);
  }, []);

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: '#D1252A',
        justifyContent: 'center',
        alignItems: 'center',
      }}>
      <StatusBar barStyle="light-content" backgroundColor="#4CAF50" />

      <Animated.View
        style={{
          opacity: fadeAnim,
          transform: [{ scale: scaleAnim }],
          alignItems: 'center',
        }}>
        {/* Logo Icon */}
        <View
          style={{
            width: 120,
            height: 120,
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
            borderRadius: 60,
            justifyContent: 'center',
            alignItems: 'center',
            marginBottom: 24,
          }}>
          <Image
            source={require('../assets/logo2.png')}
            style={{
              width: 90,
              height: 90,
              resizeMode: 'contain',
            }}
          />
        </View>

        {/* App Name */}
        <Text
          style={{
            fontSize: 42,
            fontWeight: '700',
            color: '#FFFFFF',
            letterSpacing: 1,
          }}>
          OnlyFoods
        </Text>

        <Text
          style={{
            fontSize: 16,
            color: 'rgba(255, 255, 255, 0.9)',
            marginTop: 8,
            fontWeight: '400',
          }}>
          Find your perfect meal
        </Text>
      </Animated.View>
    </View>
  );
}
