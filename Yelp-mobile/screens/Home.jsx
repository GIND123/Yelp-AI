import React, { useState, useContext } from 'react';
import { View, Text, ScrollView, TouchableOpacity, SafeAreaView, StatusBar } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { styles } from '../styles/styles';
import { useNavigation } from '@react-navigation/native';
import LoadingScreen from '../components/LoadingScreen';

import { AppContext } from '../context/AppContext';

import FoodCard from '../components/FoodCard';

const FEED_DATA = [
  {
    id: 1,
    restaurant: 'The Corner Bistro',
    date: 'June 15, 2024',
    time: '7:30 PM',
    imageUri:
      'https://images.unsplash.com/photo-1569718212165-3a8278d5f624?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
  },
  {
    id: 2,
    restaurant: 'Fresh Catch Sushi',
    date: 'June 12, 2024',
    time: '1:00 PM',
    imageUri:
      'https://images.unsplash.com/photo-1579871494447-9811cf80d66c?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
  },
  {
    id: 3,
    restaurant: 'Noodle House',
    date: 'June 10, 2024',
    time: '8:15 PM',
    imageUri:
      'https://images.unsplash.com/photo-1569718212165-3a8278d5f624?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
  },
];

export default function Home() {
  const navigation = useNavigation();

  const { isLoading, setIsLoading, business, setBusiness } = useContext(AppContext);

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="#F9F9F9" />
      {isLoading ? <LoadingScreen /> : null}
      <View style={styles.container}>
        {/* Scrollable Feed */}
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}>
          <Text style={styles.headerTitle}>Recent orders</Text>
          {FEED_DATA.map((item) => (
            <FoodCard
              key={item.id}
              imageUri={item.imageUri}
              restaurant={item.restaurant}
              date={item.date}
              time={item.time}
            />
          ))}
        </ScrollView>

        {/* Floating Action Button */}
        <View style={styles.fabContainer}>
          <TouchableOpacity
            style={styles.fabButton}
            activeOpacity={0.8}
            onPress={() => navigation.navigate('AddPhoto')}>
            {/* Icons work exactly the same way with the new import */}
            <MaterialIcons name="photo-camera" size={24} color="#FFFFFF" />
            <Text style={styles.fabText}>New Order</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}
