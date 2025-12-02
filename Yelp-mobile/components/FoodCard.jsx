import { View, Text, Image, TouchableOpacity } from 'react-native';

import MaterialIcons from 'react-native-vector-icons/MaterialIcons';

import { styles } from '../styles/styles';

const FoodCard = ({ imageUri, restaurant, date, time }) => {
  return (
    <TouchableOpacity style={styles.cardContainer} activeOpacity={0.9}>
      <View style={styles.cardImageContainer}>
        <Image source={{ uri: imageUri }} style={styles.cardImage} />
      </View>

      <View style={styles.cardContent}>
        <Text style={styles.restaurantName}>{restaurant}</Text>

        <View style={styles.metaRow}>
          <MaterialIcons name="calendar-today" size={16} color="#6B7280" />
          <Text style={styles.metaText}>{date}</Text>
          <Text style={styles.dotSeparator}>•</Text>
          <Text style={styles.metaText}>{time}</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
};

export default FoodCard;
