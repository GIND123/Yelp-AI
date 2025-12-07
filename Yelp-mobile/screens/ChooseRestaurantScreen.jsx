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
      let selectedBusiness = business?.filter((item) => item.id === selectedId);

      if (!selectedBusiness || selectedBusiness.length === 0) {
        Alert.alert('Error', 'No business data available. Please try again.');
        return;
      }

      setIsLoading(true);
      const details = await getRestaurantDetails(selectedBusiness[0]?.yelp_url);
      setIsLoading(false);

      if (!details) {
        Alert.alert('Error', 'Failed to load restaurant details. Please try again.');
        return;
      }

      navigation.navigate('RestaurantSummaryScreen', {
        data: selectedBusiness,
        details: details, // Pass the fetched details
      });
    } catch (error) {
      setIsLoading(false);
      console.error('Failed to get restaurant details:', error);
      // Show error to user
      Alert.alert('Error', 'Failed to load restaurant details. Please try again.');
      return;
    }
  }

  const renderItem = ({ item }) => {
    const isSelected = item.id === selectedId;

    return (
      <TouchableOpacity
        activeOpacity={0.7}
        onPress={() => setSelectedId(item.id)}
        style={[
          styles.restaurantCard,
          isSelected ? styles.restaurantCardSelected : styles.restaurantCardUnselected,
        ]}>
        {/* Left: Image */}
        <Image
          source={{ uri: item.photo_url }}
          style={[
            styles.restaurantThumbnail,
            isSelected && {
              borderWidth: 2,
              borderColor: COLORS.primary,
            },
          ]}
        />

        {/* Middle: Info */}
        <View style={{ flex: 1 }}>
          <Text
            style={[styles.restaurantName, isSelected && { color: COLORS.primary }]}
            numberOfLines={1}>
            {item.name}
          </Text>
          <Text style={styles.cuisineText}>{item.price}</Text>

          <View style={styles.metaRow}>
            <View style={styles.ratingContainer}>
              <MaterialIcons name="star" size={16} color={COLORS.gold} />
              <Text style={styles.ratingText}>{item.rating}</Text>
            </View>
            <Text style={{ color: '#D1D5DB', marginRight: 8 }}>•</Text>
            <Text style={styles.metaText} numberOfLines={1}>
              {item.distance}
            </Text>
          </View>
        </View>

        {/* Right: Selection Indicator */}
        <View
          style={[
            styles.selectionRadio,
            isSelected ? styles.selectionRadioSelected : styles.selectionRadioUnselected,
          ]}>
          {isSelected && <MaterialIcons name="check" size={20} color="#FFF" />}
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
      <LoadingScreen visible={isLoading} />
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
