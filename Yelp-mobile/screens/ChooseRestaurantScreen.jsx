import React, { useState, useContext } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  FlatList,
  SafeAreaView,
  StatusBar,
  Alert,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { AppContext } from '../context/AppContext';
import { useNavigation } from '@react-navigation/native';
import { getRestaurantDetails } from '../services/Endpoints';
import LoadingScreen from '../components/LoadingScreen';

import { styles, COLORS } from '../styles/styles';

export default function ChooseRestaurantScreen() {
  const { isLoading, setIsLoading, business } = useContext(AppContext);
  const [selectedId, setSelectedId] = useState('1');
  const navigation = useNavigation();

  async function onConfirmRestaurant() {
    try {
      let selectedBusiness = business.filter((item) => item.id === selectedId);
      setIsLoading(true);
      const details = await getRestaurantDetails(selectedBusiness[0].yelp_url);
      setIsLoading(false);

      navigation.navigate('RestaurantSummaryScreen', {
        data: selectedBusiness,
        details: details, // Pass the fetched details
      });
    } catch (error) {
      setIsLoading(false);
      console.error('Failed to get restaurant details:', error);
      // Show error to user
      Alert.alert('Error', 'Failed to load restaurant details. Please try again.');
    }
  }

  const renderItem = ({ item }) => {
    const isSelected = item.id === selectedId;

    return (
      <TouchableOpacity
        activeOpacity={0.9}
        onPress={() => setSelectedId(item.id)}
        style={[
          styles.restaurantCard, // Updated Name
          isSelected ? styles.restaurantCardSelected : styles.restaurantCardUnselected, // Updated Names
        ]}>
        {/* Left: Image */}
        <Image source={{ uri: item.photo_url }} style={styles.restaurantThumbnail} />

        {/* Middle: Info */}
        <View style={{ flex: 1 }}>
          <Text style={styles.restaurantName}>{item.name}</Text>
          <Text style={styles.cuisineText}>{item.price}</Text>

          <View style={styles.metaRow}>
            <View style={styles.ratingContainer}>
              <MaterialIcons name="star" size={16} color={COLORS.gold} />
              <Text style={styles.ratingText}>{item.rating}</Text>
            </View>
            <Text style={{ color: '#D1D5DB', marginRight: 8 }}>•</Text>
            <Text style={styles.metaText}>{item.distance}</Text>
          </View>
        </View>

        {/* Right: Selection Indicator */}
        <View
          style={[
            styles.selectionRadio, // Updated Name
            isSelected ? styles.selectionRadioSelected : styles.selectionRadioUnselected, // Updated Names
          ]}>
          {isSelected && <MaterialIcons name="check" size={20} color="#FFF" />}
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.backgroundLight} />
      {isLoading ? <LoadingScreen /> : null}
      {/* List */}
      <FlatList
        data={business}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.restaurantListContent} // Updated Name
        showsVerticalScrollIndicator={false}
        style={{ width: '100%' }} // Ensure list takes full width
      />

      {/* Bottom Action Button */}
      <View style={styles.bottomFixedFooter}>
        <TouchableOpacity
          style={[styles.btnBase, styles.btnPrimary]}
          onPress={() => onConfirmRestaurant()}>
          <Text style={styles.btnTextPrimary}>Confirm Restaurant</Text>
          <MaterialIcons name="arrow-forward" size={24} color="#FFF" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}
