FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Expose port 3000
EXPOSE 3000

# Start dev server
CMD ["npm", "run", "dev"]
